"""Unified Evaluation Script for all novel detection architectures.

Usage:
    python test.py --arch otdet    --data path/to/dataset --weights runs/otdet/best.pt
    python test.py --arch wavedet  --data path/to/dataset --weights runs/wavedet/best.pt
    python test.py --arch scalenet --data path/to/dataset --weights runs/scalenet/best.pt
"""

import argparse
import os
import sys

# Src-layout: add src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from data.dataset import YOLOOBBDataset, collate_fn
from utils.metrics import compute_map, compute_ap
from utils.checkpoint import safe_load_checkpoint
from utils.factory import build_model


def _build_model(arch: str, cfg: dict, img_size: int, device: torch.device):
    """Construct model using shared factory (pretrained=False for eval)."""
    import copy

    cfg = copy.deepcopy(cfg)
    # Only OTDet uses a pretrained backbone; disable to avoid downloading
    # weights when loading from a checkpoint
    if arch == "otdet":
        cfg["model"]["pretrained"] = False
    return build_model(arch, cfg, img_size, device)


def parse_args():
    p = argparse.ArgumentParser(description="Novel Detection — Evaluation")
    p.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["otdet", "wavedet", "scalenet", "toponet", "flownet", "infogeonet"],
        help="Architecture (auto-detected from checkpoint if not given)",
    )
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def evaluate():
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    ckpt = safe_load_checkpoint(args.weights, map_location=device)
    cfg = ckpt.get("config", {})
    arch = args.arch or ckpt.get("arch", "otdet")

    if not cfg:
        config_map = {
            "otdet": "configs/otdet.yaml",
            "wavedet": "configs/wavedet.yaml",
            "scalenet": "configs/scalenet.yaml",
            "toponet": "configs/toponet.yaml",
            "flownet": "configs/flownet.yaml",
            "infogeonet": "configs/infogeonet.yaml",
        }
        with open(config_map[arch], "r") as f:
            cfg = yaml.safe_load(f)

    img_size = cfg["data"]["img_size"]
    nc = cfg["model"]["num_classes"]

    dataset = YOLOOBBDataset(
        root=args.data, split=args.split, img_size=img_size, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"[{arch.upper()}] Evaluating {len(dataset)} images ({args.split})")

    model = _build_model(arch, cfg, img_size, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            results = model.inference(images.to(device), args.conf, args.nms)
            for r, t in zip(results, targets):
                all_preds.append(r)
                all_targets.append(
                    {"obbs": t["obbs"].numpy(), "classes": t["classes"].numpy()}
                )

    thresholds = cfg.get("eval", {}).get("iou_thresholds", [0.5])
    metrics = compute_map(all_preds, all_targets, thresholds, nc)

    print(f"\n{'='*50}")
    print(f"  {arch.upper()} Evaluation Results")
    print(f"{'='*50}")
    print(f"  mAP: {metrics['map']:.4f}")
    for t, v in metrics["map_per_thresh"].items():
        print(f"  mAP@{t}: {v:.4f}")
    print(f"{'='*50}")

    return metrics


if __name__ == "__main__":
    evaluate()
