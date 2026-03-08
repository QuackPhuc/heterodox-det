"""Unified dry-run smoke test for all 6 novel architectures.

Usage:
    python scripts/dry_run.py                      # test all 6
    python scripts/dry_run.py --arch otdet         # test specific one
    python scripts/dry_run.py --arch wavedet
    python scripts/dry_run.py --arch scalenet
    python scripts/dry_run.py --arch toponet
    python scripts/dry_run.py --arch flownet
    python scripts/dry_run.py --arch infogeonet
"""

import os
import sys
import tempfile
import shutil
import random
import argparse

import numpy as np
import torch

# Src-layout: add src/ to import path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from models import OTDet, WaveDetNet, ScaleNet, TopoNet, FlowNet, InfoGeoNet
from data.dataset import YOLOOBBDataset, collate_fn
from losses import OTDetLoss, PeakDetLoss


def create_synthetic_dataset(root, num_images=16, img_size=320, num_classes=5):
    import cv2

    for split in ["train", "val"]:
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n = num_images if split == "train" else max(4, num_images // 4)
        for i in range(n):
            img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(img_dir, f"img_{i:04d}.jpg"), img)
            num_objs = random.randint(1, 5)
            with open(os.path.join(lbl_dir, f"img_{i:04d}.txt"), "w") as f:
                for _ in range(num_objs):
                    cls_id = random.randint(0, num_classes - 1)
                    cx, cy = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                    w, h = random.uniform(0.05, 0.3), random.uniform(0.05, 0.3)
                    angle = random.uniform(-0.5, 0.5)
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    dx, dy = w / 2, h / 2
                    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
                    rotated = []
                    for x, y in corners:
                        rotated.extend(
                            [
                                max(0, min(1, x * cos_a - y * sin_a + cx)),
                                max(0, min(1, x * sin_a + y * cos_a + cy)),
                            ]
                        )
                    f.write(f"{cls_id} {' '.join(f'{v:.6f}' for v in rotated)}\n")


def test_arch(name, model, criterion, loader, device, img_size, num_steps=5):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for step, (images, targets) in enumerate(loader):
        if step >= num_steps:
            break
        images = images.to(device)
        preds = model(images)

        if step == 0:
            print(f"  Shapes: ", end="")
            shapes = {
                k: tuple(v.shape)
                for k, v in preds.items()
                if isinstance(v, torch.Tensor)
            }
            print(", ".join(f"{k}={s}" for k, s in list(shapes.items())[:4]))

        loss_dict = criterion(preds, targets, img_size=img_size)
        loss = loss_dict["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Step {step+1}: loss={loss.item():.4f}")

    # Inference
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        results = model.inference(images.to(device), conf_thresh=0.1)
        total = sum(len(r["obbs"]) for r in results)
        print(f"  Inference: {total} detections")

    # Gradient check
    model.train()
    images, targets = next(iter(loader))
    preds = model(images.to(device))
    criterion(preds, targets, img_size=img_size)["total"].backward()
    ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"  Gradients: {'✅' if ok else '❌'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        default="all",
        choices=[
            "all",
            "otdet",
            "wavedet",
            "scalenet",
            "toponet",
            "flownet",
            "infogeonet",
        ],
    )
    args = parser.parse_args()

    nc, img_size, bs = 5, 320, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"  6 Novel Detection Architectures — Smoke Test")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    tmp = tempfile.mkdtemp(prefix="novel_det_")
    create_synthetic_dataset(tmp, 16, img_size, nc)
    ds = YOLOOBBDataset(root=tmp, split="train", img_size=img_size, augment=False)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    archs = {}
    ot_loss = OTDetLoss(num_classes=nc)
    pk_loss = PeakDetLoss(num_classes=nc)

    if args.arch in ("all", "otdet"):
        archs["① OTDet (Optimal Transport)"] = (
            OTDet(
                num_classes=nc,
                num_slots=20,
                feat_channels=128,
                pretrained_backbone=False,
                sinkhorn_iters=10,
                img_size=img_size,
            ).to(device),
            ot_loss,
        )
    if args.arch in ("all", "wavedet"):
        archs["② WaveDetNet (Wave Resonance)"] = (
            WaveDetNet(
                num_classes=nc,
                feat_channels=128,
                num_proposals=20,
                num_wave_steps=6,
                img_size=img_size,
            ).to(device),
            pk_loss,
        )
    if args.arch in ("all", "scalenet"):
        archs["③ ScaleNet (Continuous Scale-Space)"] = (
            ScaleNet(
                num_classes=nc,
                feat_channels=128,
                num_proposals=20,
                num_scales=6,
                sigma_range=(0.5, 4.0),
                img_size=img_size,
            ).to(device),
            pk_loss,
        )
    if args.arch in ("all", "toponet"):
        archs["④ TopoNet (Topological Persistence)"] = (
            TopoNet(
                num_classes=nc,
                feat_channels=128,
                num_proposals=20,
                num_filtration_steps=12,
                img_size=img_size,
            ).to(device),
            pk_loss,
        )
    if args.arch in ("all", "flownet"):
        archs["⑤ FlowNet (Neural ODE Attractors)"] = (
            FlowNet(
                num_classes=nc,
                feat_channels=128,
                num_proposals=20,
                ode_steps=6,
                img_size=img_size,
            ).to(device),
            pk_loss,
        )

    if args.arch in ("all", "infogeonet"):
        archs["⑥ InfoGeoNet (Fisher Information)"] = (
            InfoGeoNet(
                num_classes=nc,
                feat_channels=128,
                num_proposals=20,
                num_fisher_samples=0,
                img_size=img_size,
            ).to(device),
            pk_loss,
        )

    results = {}
    try:
        for name, (model, crit) in archs.items():
            results[name] = test_arch(name, model, crit, dl, device, img_size)

        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        for name, ok in results.items():
            print(f"  {name:45s} {'✅ PASS' if ok else '❌ FAIL'}")
        print(f"{'='*60}")
        if all(results.values()):
            print(f"\n  🎉 ALL 6 ARCHITECTURES PASSED!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
