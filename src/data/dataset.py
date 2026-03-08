"""YOLO OBB v5 format dataset loader.

Expected directory structure:
    dataset_root/
        images/
            train/
                img001.jpg
                ...
            val/
                ...
        labels/
            train/
                img001.txt
                ...
            val/
                ...

Label format per line:  class_id x1 y1 x2 y2 x3 y3 x4 y4
Coordinates are normalized to [0, 1].
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.obb_utils import poly_to_obb


class YOLOOBBDataset(Dataset):
    """Dataset for YOLO Oriented Object Detection v5 format."""

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        augment: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        flip_lr: float = 0.5,
        flip_ud: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud

        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        self.img_paths = sorted(
            p
            for p in glob.glob(os.path.join(img_dir, "*"))
            if os.path.splitext(p)[1].lower() in self.IMG_EXTENSIONS
        )
        self.lbl_dir = lbl_dir

        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {img_dir}. "
                "Ensure the dataset follows YOLO OBB v5 structure."
            )

    def __len__(self):
        return len(self.img_paths)

    def _load_label(self, img_path: str) -> np.ndarray:
        """Load label file corresponding to an image.

        Returns:
            labels: (N, 9) — class_id, x1, y1, ..., x4, y4 (normalized)
        """
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, stem + ".txt")

        if not os.path.exists(lbl_path):
            return np.zeros((0, 9), dtype=np.float32)

        labels = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                vals = [float(x) for x in parts[:9]]
                labels.append(vals)

        if len(labels) == 0:
            return np.zeros((0, 9), dtype=np.float32)
        return np.array(labels, dtype=np.float32)

    def _augment_hsv(self, img: np.ndarray):
        """Random HSV augmentation."""
        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        img_hsv = cv2.merge(
            [
                cv2.LUT(hue, lut_hue),
                cv2.LUT(sat, lut_sat),
                cv2.LUT(val, lut_val),
            ]
        )
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # Load labels (normalized coordinates)
        labels = self._load_label(img_path)

        # Resize to square
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Augmentations
        if self.augment:
            img = self._augment_hsv(img)

            # Horizontal flip
            if random.random() < self.flip_lr:
                img = np.fliplr(img).copy()
                if len(labels) > 0:
                    # Flip x coordinates: x' = 1 - x
                    labels[:, 1::2] = 1.0 - labels[:, 1::2]

            # Vertical flip
            if random.random() < self.flip_ud:
                img = np.flipud(img).copy()
                if len(labels) > 0:
                    # Flip y coordinates: y' = 1 - y
                    labels[:, 2::2] = 1.0 - labels[:, 2::2]

        # Convert to tensor (B, C, H, W), float [0, 1]
        img = img[:, :, ::-1].copy()  # BGR → RGB
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img)

        # Build target dict
        target = self._build_target(labels)
        return img_tensor, target

    def _build_target(self, labels: np.ndarray) -> dict:
        """Convert raw labels to target dict.

        Returns dict with:
            'classes': (N,) int64
            'obbs':    (N, 5) float32 — cx, cy, w, h, angle (absolute pixels at img_size)
            'polys':   (N, 8) float32 — polygon (absolute pixels)
        """
        if len(labels) == 0:
            return {
                "classes": torch.zeros(0, dtype=torch.int64),
                "obbs": torch.zeros((0, 5), dtype=torch.float32),
                "polys": torch.zeros((0, 8), dtype=torch.float32),
            }

        classes = labels[:, 0].astype(np.int64)

        # Denormalize polygon coordinates to img_size
        polys = labels[:, 1:].copy()
        polys[:, 0::2] *= self.img_size  # x coordinates
        polys[:, 1::2] *= self.img_size  # y coordinates

        # Convert polygons to OBBs
        obbs = poly_to_obb(polys)

        return {
            "classes": torch.from_numpy(classes),
            "obbs": torch.from_numpy(obbs),
            "polys": torch.from_numpy(polys.astype(np.float32)),
        }


def collate_fn(batch):
    """Custom collate to handle variable number of targets per image."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
