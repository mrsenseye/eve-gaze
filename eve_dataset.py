"""
EVE Dataset PyTorch DataLoader

Loads face and eye videos with gaze labels from the EVE dataset.
Dataset: https://ait.ethz.ch/eve

Data structure:
    eve_dataset/
        train01-39/
        val01-05/
        test01-10/
            step*_image_*/
                webcam_c_face.mp4   # 256x256 face crops at 30fps
                webcam_c_eyes.mp4   # 256x128 both eyes side-by-side (left:0-128, right:128-256)
                webcam_c.h5         # Gaze labels and metadata
                ...

Patch types:
    - face: 256x256 face crops
    - left: 128x128 left eye crop (extracted from eyes video)
    - right: 128x128 right eye crop (extracted from eyes video)

Note: Test split does not include ground truth gaze labels (face_g_tobii, face_PoG_tobii).
      For test split, use target_type='head' to load head pose, or labels will be NaN.
"""

import os
from pathlib import Path
from typing import Optional, Literal

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def get_gaze_augmentations(p: float = 0.5):
    """
    Get augmentations safe for gaze estimation.

    NO horizontal/vertical flip or rotation - these would change gaze direction.
    Only color/intensity augmentations that don't affect spatial gaze info.
    """
    return T.Compose([
        T.RandomApply([
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.1,
            )
        ], p=p),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.2),
    ])


class SynchronizedAugmentation:
    """
    Augmentation that applies the SAME random transformation to multiple images.

    Critical for multi-stream: face and eyes must get identical color changes.
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.color_jitter = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
        )

    def __call__(self, frames_dict: dict) -> dict:
        """
        Apply synchronized augmentation to all streams.

        Args:
            frames_dict: Dict of tensors {'face': ..., 'left': ..., 'right': ...}

        Returns:
            Augmented frames dict with same transforms applied to all streams
        """
        # Decide which augmentations to apply (same for all streams)
        apply_color = torch.rand(1).item() < self.p
        apply_gray = torch.rand(1).item() < 0.1
        apply_blur = torch.rand(1).item() < 0.2

        result = {}

        for key, frames in frames_dict.items():
            aug_frames = frames

            if apply_color and not apply_gray:
                # Get color jitter parameters (same for all streams)
                if key == list(frames_dict.keys())[0]:  # First stream: generate params
                    # Generate random parameters
                    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                        T.ColorJitter.get_params(
                            self.color_jitter.brightness,
                            self.color_jitter.contrast,
                            self.color_jitter.saturation,
                            self.color_jitter.hue,
                        )
                    self._color_params = (fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)

                # Apply same parameters to all streams
                fn_idx, bf, cf, sf, hf = self._color_params
                for fn_id in fn_idx:
                    if fn_id == 0 and bf is not None:
                        aug_frames = T.functional.adjust_brightness(aug_frames, bf)
                    elif fn_id == 1 and cf is not None:
                        aug_frames = T.functional.adjust_contrast(aug_frames, cf)
                    elif fn_id == 2 and sf is not None:
                        aug_frames = T.functional.adjust_saturation(aug_frames, sf)
                    elif fn_id == 3 and hf is not None:
                        aug_frames = T.functional.adjust_hue(aug_frames, hf)

            if apply_gray:
                aug_frames = T.functional.rgb_to_grayscale(aug_frames, num_output_channels=3)

            if apply_blur:
                # Same blur for all streams
                if key == list(frames_dict.keys())[0]:
                    self._blur_sigma = 0.1 + torch.rand(1).item() * 1.9  # sigma in [0.1, 2.0]
                kernel_size = 5
                aug_frames = T.functional.gaussian_blur(aug_frames, kernel_size, [self._blur_sigma, self._blur_sigma])

            result[key] = aug_frames

        return result


class EVEDataset(Dataset):
    """
    PyTorch Dataset for the EVE gaze estimation dataset.

    Args:
        root_dir: Path to eve_dataset directory
        split: One of 'train', 'val', or 'test'
        camera: Which camera to use:
            - 'c', 'l', 'r': Webcams (center/left/right) at 30fps
            - 'basler': High-quality industrial camera at 60fps (recommended for iPhone inference)
        sequence_length: Number of frames per sample (1 for single frames)
        stride: Stride between sequences when sequence_length > 1
        transform: Optional transform to apply to frames
        target_type: Type of target ('PoG' for point of gaze, 'gaze' for direction, 'head' for head pose, 'both' for gaze+PoG)
        patch_types: List of patches to load: 'face', 'left', 'right'. Default ['face'] for backwards compat.
        face_size: Output size for face patches (default 224 for ViT)
        eye_size: Output size for eye patches (default 112, half of face)
    """

    SPLIT_PREFIXES = {
        'train': 'train',
        'val': 'val',
        'test': 'test',
    }

    def __init__(
        self,
        root_dir: str,
        split: Literal['train', 'val', 'test'] = 'train',
        camera: Literal['c', 'l', 'r', 'basler'] = 'c',
        sequence_length: int = 1,
        stride: int = 1,
        transform=None,
        target_type: Literal['PoG', 'gaze', 'head', 'both', 'all'] = 'gaze',
        image_size: int = 256,  # Deprecated, use face_size instead
        face_size: int = None,  # Output size for face patches (224 for ViT)
        eye_size: int = None,  # Output size for eye patches (112 default)
        patch_types: list[str] = None,  # ['face'], ['face', 'left', 'right'], etc.
        augment: bool = False,  # Apply color/blur augmentations (train only)
        load_segmentation: bool = False,  # Load per-eye segmentation masks
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.camera = camera
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.target_type = target_type
        self.augment = augment and split == 'train'  # Only augment training data
        self.load_segmentation = load_segmentation

        # Handle patch_types
        if patch_types is None:
            self.patch_types = ['face']
        else:
            valid_patches = {'face', 'left', 'right'}
            for p in patch_types:
                if p not in valid_patches:
                    raise ValueError(f"Invalid patch type '{p}'. Valid: {valid_patches}")
            self.patch_types = list(patch_types)

        # Handle image sizes (backwards compat with image_size)
        if face_size is not None:
            self.face_size = face_size
        elif image_size != 256:
            self.face_size = image_size  # Use old image_size param
        else:
            self.face_size = 224  # Default for ViT

        if eye_size is not None:
            self.eye_size = eye_size
        else:
            self.eye_size = self.face_size // 2  # Default: half of face size

        # Keep image_size for backwards compatibility
        self.image_size = self.face_size

        # Flags for which video files we need
        self.load_face = 'face' in self.patch_types
        self.load_eyes = 'left' in self.patch_types or 'right' in self.patch_types
        self.is_multistream = len(self.patch_types) > 1

        # Setup augmentation pipeline
        if self.augment:
            if self.is_multistream:
                # Use synchronized augmentation for multi-stream
                self.sync_aug = SynchronizedAugmentation(p=0.5)
                self.aug_transform = None  # Don't use per-stream augmentation
            else:
                # Single stream: use standard augmentation
                self.sync_aug = None
                self.aug_transform = get_gaze_augmentations(p=0.5)
        else:
            self.aug_transform = None
            self.sync_aug = None

        # Build index of all samples
        self.samples = self._build_index()

    def _build_index(self) -> list[dict]:
        """Build an index of all valid samples in the dataset."""
        samples = []
        prefix = self.SPLIT_PREFIXES[self.split]

        # Find all participant folders for this split
        participant_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ])

        for participant_dir in participant_dirs:
            # Find all step folders
            step_dirs = sorted([
                d for d in participant_dir.iterdir()
                if d.is_dir() and d.name.startswith('step')
            ])

            for step_dir in step_dirs:
                # Handle basler vs webcam file naming
                if self.camera == 'basler':
                    face_video_path = step_dir / 'basler_face.mp4'
                    eyes_video_path = step_dir / 'basler_eyes.mp4'
                    h5_path = step_dir / 'basler.h5'
                else:
                    face_video_path = step_dir / f'webcam_{self.camera}_face.mp4'
                    eyes_video_path = step_dir / f'webcam_{self.camera}_eyes.mp4'
                    h5_path = step_dir / f'webcam_{self.camera}.h5'

                # Check required files exist
                if not h5_path.exists():
                    continue
                if self.load_face and not face_video_path.exists():
                    continue
                if self.load_eyes and not eyes_video_path.exists():
                    continue

                # Get number of frames and valid indices
                with h5py.File(h5_path, 'r') as f:
                    if self.target_type == 'PoG':
                        if 'face_PoG_tobii/validity' not in f:
                            continue  # Skip if no gaze labels (test set)
                        validity = f['face_PoG_tobii/validity'][:]
                    elif self.target_type == 'gaze':
                        if 'face_g_tobii/validity' not in f:
                            continue  # Skip if no gaze labels (test set)
                        validity = f['face_g_tobii/validity'][:]
                    elif self.target_type == 'both':
                        # Need both gaze and PoG to be valid
                        if 'face_g_tobii/validity' not in f or 'face_PoG_tobii/validity' not in f:
                            continue  # Skip if no labels (test set)
                        gaze_validity = f['face_g_tobii/validity'][:]
                        pog_validity = f['face_PoG_tobii/validity'][:]
                        validity = gaze_validity & pog_validity  # Both must be valid
                    elif self.target_type == 'all':
                        # Need gaze, PoG, pupil, and head pose to be valid
                        if 'face_g_tobii/validity' not in f or 'face_PoG_tobii/validity' not in f:
                            continue  # Skip if no labels (test set)
                        if 'left_p/validity' not in f or 'right_p/validity' not in f:
                            continue  # Skip if no pupil data
                        gaze_validity = f['face_g_tobii/validity'][:]
                        pog_validity = f['face_PoG_tobii/validity'][:]
                        left_pupil_validity = f['left_p/validity'][:]
                        right_pupil_validity = f['right_p/validity'][:]
                        head_validity = f['face_h/validity'][:]
                        validity = gaze_validity & pog_validity & left_pupil_validity & right_pupil_validity & head_validity
                    else:  # head pose (available in all splits)
                        validity = f['face_h/validity'][:]
                    num_frames = len(validity)

                # Create samples based on sequence length
                if self.sequence_length == 1:
                    # Single frame mode - one sample per valid frame
                    for idx in range(num_frames):
                        if validity[idx]:
                            samples.append({
                                'face_video_path': str(face_video_path) if self.load_face else None,
                                'eyes_video_path': str(eyes_video_path) if self.load_eyes else None,
                                'h5_path': str(h5_path),
                                'frame_indices': [idx],
                                'participant': participant_dir.name,
                                'step': step_dir.name,
                            })
                else:
                    # Sequence mode - sliding window
                    for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                        end_idx = start_idx + self.sequence_length
                        # Check if all frames in sequence are valid
                        if validity[start_idx:end_idx].all():
                            samples.append({
                                'face_video_path': str(face_video_path) if self.load_face else None,
                                'eyes_video_path': str(eyes_video_path) if self.load_eyes else None,
                                'h5_path': str(h5_path),
                                'frame_indices': list(range(start_idx, end_idx)),
                                'participant': participant_dir.name,
                                'step': step_dir.name,
                            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        frame_indices = sample_info['frame_indices']

        # Load frames from each required video
        frames_dict = {}

        if self.load_face:
            face_frames = self._load_face_frames(
                sample_info['face_video_path'],
                frame_indices
            )
            frames_dict['face'] = self._process_frames(face_frames, self.face_size)

        if self.load_eyes:
            left_frames, right_frames = self._load_eye_frames(
                sample_info['eyes_video_path'],
                frame_indices
            )
            if 'left' in self.patch_types:
                frames_dict['left'] = self._process_frames(left_frames, self.eye_size)
            if 'right' in self.patch_types:
                frames_dict['right'] = self._process_frames(right_frames, self.eye_size)

        # Load labels from h5
        labels = self._load_labels(
            sample_info['h5_path'],
            frame_indices
        )

        # Apply synchronized augmentation for multi-stream
        if self.is_multistream and self.sync_aug is not None:
            frames_dict = self.sync_aug(frames_dict)

        # Build result dictionary
        # For backwards compatibility: if only 'face', put it in 'frames' key
        if self.patch_types == ['face']:
            frames = frames_dict['face']
        else:
            frames = frames_dict  # Return dict of all patches

        result = {
            'frames': frames,
            'gaze': torch.from_numpy(labels['gaze']).float(),
            'gaze_validity': torch.from_numpy(labels['validity']),
            'participant': sample_info['participant'],
            'step': sample_info['step'],
        }
        # Add PoG if available (target_type='both' or 'all')
        if 'pog' in labels:
            result['pog'] = torch.from_numpy(labels['pog']).float()
            result['pog_validity'] = torch.from_numpy(labels['pog_validity'])
        # Add pupil if available (target_type='all')
        if 'pupil' in labels:
            result['pupil'] = torch.from_numpy(labels['pupil']).float()
            result['pupil_validity'] = torch.from_numpy(labels['pupil_validity'])
        # Add head pose if available (target_type='all')
        if 'head' in labels:
            result['head'] = torch.from_numpy(labels['head']).float()
            result['head_validity'] = torch.from_numpy(labels['head_validity'])

        # Load segmentation masks if enabled
        if self.load_segmentation:
            seg_masks = self._load_segmentation_masks(
                sample_info['h5_path'],
                frame_indices
            )
            if seg_masks is not None:
                result['seg_left'] = torch.from_numpy(seg_masks['left']).long()
                result['seg_right'] = torch.from_numpy(seg_masks['right']).long()
                result['seg_valid'] = torch.tensor(True)
            else:
                # No segmentation available - return zeros
                if self.sequence_length == 1:
                    result['seg_left'] = torch.zeros(self.eye_size, self.eye_size, dtype=torch.long)
                    result['seg_right'] = torch.zeros(self.eye_size, self.eye_size, dtype=torch.long)
                else:
                    result['seg_left'] = torch.zeros(self.sequence_length, self.eye_size, self.eye_size, dtype=torch.long)
                    result['seg_right'] = torch.zeros(self.sequence_length, self.eye_size, self.eye_size, dtype=torch.long)
                result['seg_valid'] = torch.tensor(False)

        return result

    def _load_face_frames(self, video_path: str, frame_indices: list[int]) -> np.ndarray:
        """Load face frames from face video (256x256)."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((256, 256, 3), dtype=np.uint8))

        cap.release()

        if len(frames) == 1:
            return frames[0]
        return np.stack(frames)

    def _load_eye_frames(self, video_path: str, frame_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Load and split eye frames from eyes video (256x128 -> left 128x128, right 128x128)."""
        cap = cv2.VideoCapture(video_path)
        left_frames = []
        right_frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Eyes video is 256x128: left eye is 0:128, right eye is 128:256
                left_eye = frame[:, :128, :]   # Left half (128x128)
                right_eye = frame[:, 128:, :]  # Right half (128x128)
                left_frames.append(left_eye)
                right_frames.append(right_eye)
            else:
                left_frames.append(np.zeros((128, 128, 3), dtype=np.uint8))
                right_frames.append(np.zeros((128, 128, 3), dtype=np.uint8))

        cap.release()

        if len(left_frames) == 1:
            return left_frames[0], right_frames[0]
        return np.stack(left_frames), np.stack(right_frames)

    def _process_frames(self, frames: np.ndarray, target_size: int) -> torch.Tensor:
        """Resize frames, convert to tensor, and apply augmentations."""
        # Resize if needed
        if frames.ndim == 3:  # Single frame (H, W, C)
            h, w = frames.shape[:2]
            if h != target_size or w != target_size:
                frames = cv2.resize(frames, (target_size, target_size),
                                    interpolation=cv2.INTER_LINEAR)
        else:  # Sequence (T, H, W, C)
            h, w = frames.shape[1:3]
            if h != target_size or w != target_size:
                resized = []
                for f in frames:
                    resized.append(cv2.resize(f, (target_size, target_size),
                                              interpolation=cv2.INTER_LINEAR))
                frames = np.stack(resized)

        # Apply custom transform if provided
        if self.transform is not None:
            frames = self.transform(frames)
        else:
            # Default: convert to tensor, normalize to [0, 1], CHW format
            frames = torch.from_numpy(frames).float() / 255.0
            if frames.ndim == 3:  # Single frame: HWC -> CHW
                frames = frames.permute(2, 0, 1)
            else:  # Sequence: THWC -> TCHW
                frames = frames.permute(0, 3, 1, 2)

        # Apply augmentations (color jitter, grayscale, blur - no flips!)
        if self.aug_transform is not None:
            frames = self.aug_transform(frames)

        return frames

    def _load_labels(self, h5_path: str, frame_indices: list[int]) -> dict:
        """Load gaze/head labels from h5 file."""
        with h5py.File(h5_path, 'r') as f:
            if self.target_type == 'PoG':
                # Point of gaze in screen coordinates (pixels)
                gaze = f['face_PoG_tobii/data'][frame_indices]
                validity = f['face_PoG_tobii/validity'][frame_indices]
                pog = None
                pog_validity = None
            elif self.target_type == 'gaze':
                # Gaze direction (pitch, yaw in radians)
                gaze = f['face_g_tobii/data'][frame_indices]
                validity = f['face_g_tobii/validity'][frame_indices]
                pog = None
                pog_validity = None
            elif self.target_type == 'both':
                # Load both gaze and PoG
                gaze = f['face_g_tobii/data'][frame_indices]
                validity = f['face_g_tobii/validity'][frame_indices]
                pog = f['face_PoG_tobii/data'][frame_indices]
                pog_validity = f['face_PoG_tobii/validity'][frame_indices]
            elif self.target_type == 'all':
                # Load gaze, PoG, pupil, and head pose
                gaze = f['face_g_tobii/data'][frame_indices]
                validity = f['face_g_tobii/validity'][frame_indices]
                pog = f['face_PoG_tobii/data'][frame_indices]
                pog_validity = f['face_PoG_tobii/validity'][frame_indices]
                left_pupil = f['left_p/data'][frame_indices]
                right_pupil = f['right_p/data'][frame_indices]
                left_pupil_validity = f['left_p/validity'][frame_indices]
                right_pupil_validity = f['right_p/validity'][frame_indices]
                head_pose = f['face_h/data'][frame_indices]
                head_validity = f['face_h/validity'][frame_indices]
            else:  # head pose
                # Head pose (pitch, yaw in radians)
                gaze = f['face_h/data'][frame_indices]
                validity = f['face_h/validity'][frame_indices]
                pog = None
                pog_validity = None

        # Squeeze if single frame
        if len(frame_indices) == 1:
            gaze = gaze[0]
            validity = validity[0:1]  # Keep as array for consistency
            if pog is not None:
                pog = pog[0]
                pog_validity = pog_validity[0:1]
            if self.target_type == 'all':
                left_pupil = left_pupil[0:1]  # Keep as array, shape (1,)
                right_pupil = right_pupil[0:1]
                left_pupil_validity = left_pupil_validity[0:1]
                right_pupil_validity = right_pupil_validity[0:1]
                head_pose = head_pose[0]
                head_validity = head_validity[0:1]

        result = {'gaze': gaze, 'validity': validity}
        if pog is not None:
            result['pog'] = pog
            result['pog_validity'] = pog_validity
        if self.target_type == 'all':
            # Stack left and right pupil as (T, 2) or (2,) for single frame
            result['pupil'] = np.stack([left_pupil, right_pupil], axis=-1)  # (T, 2) or (1, 2)
            result['pupil_validity'] = left_pupil_validity & right_pupil_validity
            result['head'] = head_pose
            result['head_validity'] = head_validity
        return result

    def _load_segmentation_masks(
        self,
        h5_path: str,
        frame_indices: list[int]
    ) -> Optional[dict]:
        """
        Load per-eye segmentation masks from h5 file.

        Returns:
            Dict with 'left' and 'right' masks, or None if not available.
            Masks are at eye_size resolution (112x112 by default).
        """
        try:
            with h5py.File(h5_path, 'r') as f:
                # Check if eye segmentation masks exist
                if 'segmentation/masks_left_hybrid' not in f or 'segmentation/masks_right_hybrid' not in f:
                    return None

                left_masks = f['segmentation/masks_left_hybrid'][frame_indices]
                right_masks = f['segmentation/masks_right_hybrid'][frame_indices]

                # Remap left eye classes: 4→1 (sclera), 5→2 (iris), 6→3 (pupil)
                # The face segmentation model outputs: 0=bg, 1-3=right eye, 4-6=left eye
                # We need unified classes: 0=bg, 1=sclera, 2=iris, 3=pupil
                left_masks = np.where(left_masks == 4, 1, left_masks)
                left_masks = np.where(left_masks == 5, 2, left_masks)
                left_masks = np.where(left_masks == 6, 3, left_masks)

                # Resize if needed (stored at 112x112, may need different size)
                stored_size = left_masks.shape[-1]
                if stored_size != self.eye_size:
                    if left_masks.ndim == 2:  # Single frame
                        left_masks = cv2.resize(
                            left_masks, (self.eye_size, self.eye_size),
                            interpolation=cv2.INTER_NEAREST
                        )
                        right_masks = cv2.resize(
                            right_masks, (self.eye_size, self.eye_size),
                            interpolation=cv2.INTER_NEAREST
                        )
                    else:  # Sequence
                        left_resized = []
                        right_resized = []
                        for i in range(len(left_masks)):
                            left_resized.append(cv2.resize(
                                left_masks[i], (self.eye_size, self.eye_size),
                                interpolation=cv2.INTER_NEAREST
                            ))
                            right_resized.append(cv2.resize(
                                right_masks[i], (self.eye_size, self.eye_size),
                                interpolation=cv2.INTER_NEAREST
                            ))
                        left_masks = np.stack(left_resized)
                        right_masks = np.stack(right_resized)

                # Squeeze if single frame
                if len(frame_indices) == 1:
                    left_masks = left_masks[0]
                    right_masks = right_masks[0]

                return {'left': left_masks, 'right': right_masks}

        except Exception:
            return None


def create_eve_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    sequence_length: int = 1,
    num_workers: int = 4,
    camera: str = 'c',
    target_type: str = 'gaze',
    transform=None,
) -> dict[str, DataLoader]:
    """
    Create train/val/test dataloaders for EVE dataset.

    Args:
        root_dir: Path to eve_dataset directory
        batch_size: Batch size for all loaders
        sequence_length: Frames per sample
        num_workers: Number of data loading workers
        camera: Which webcam ('c', 'l', 'r')
        target_type: 'PoG' or 'gaze'
        transform: Optional transform for frames

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoader instances
    """
    loaders = {}

    for split in ['train', 'val', 'test']:
        dataset = EVEDataset(
            root_dir=root_dir,
            split=split,
            camera=camera,
            sequence_length=sequence_length,
            transform=transform,
            target_type=target_type,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    return loaders


if __name__ == '__main__':
    # Quick test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/media/nas1/saw/eve_dataset')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--multi-stream', action='store_true', help='Load all patches (face, left, right)')
    args = parser.parse_args()

    if args.multi_stream:
        # Multi-stream mode: load face + both eyes
        dataset = EVEDataset(
            root_dir=args.root,
            split=args.split,
            patch_types=['face', 'left', 'right'],
            face_size=224,
            eye_size=112,
        )
        print(f"Multi-stream dataset size: {len(dataset)} samples")

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        batch = next(iter(loader))

        print(f"Patch types: {list(batch['frames'].keys())}")
        print(f"Face shape: {batch['frames']['face'].shape}")
        print(f"Left eye shape: {batch['frames']['left'].shape}")
        print(f"Right eye shape: {batch['frames']['right'].shape}")
        print(f"Gaze shape: {batch['gaze'].shape}")
        print(f"Gaze range: [{batch['gaze'].min():.3f}, {batch['gaze'].max():.3f}]")
    else:
        # Single-stream mode (backwards compatible)
        dataset = EVEDataset(root_dir=args.root, split=args.split)
        print(f"Dataset size: {len(dataset)} samples")

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        batch = next(iter(loader))

        print(f"Batch frames shape: {batch['frames'].shape}")
        print(f"Batch gaze shape: {batch['gaze'].shape}")
        print(f"Gaze range: [{batch['gaze'].min():.3f}, {batch['gaze'].max():.3f}]")
