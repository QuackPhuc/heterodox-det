"""Unified Training Script — supports DDP multi-GPU.

Single GPU:
    python train.py --arch otdet --data path/to/dataset

Multi-GPU DDP:
    torchrun --nproc_per_node=2 train.py --arch otdet --data path/to/dataset
    torchrun --nproc_per_node=4 train.py --arch wavedet --data path/to/dataset
"""

import argparse
import os
import warnings
import sys

# Src-layout: add src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.dataset import YOLOOBBDataset, collate_fn
from utils.checkpoint import safe_load_checkpoint
from utils.factory import build_model, build_loss
from utils.logger import ExperimentLogger


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return get_rank() == 0


def setup_ddp():
    """Initialize DDP process group if launched via torchrun.

    Returns:
        (device, local_rank) if DDP is active, else (None, None).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank
    return None, None


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Model factory (delegates to shared utils.factory)
# ---------------------------------------------------------------------------


def _get_model_and_loss(arch: str, cfg: dict, img_size: int, device: torch.device):
    model = build_model(arch, cfg, img_size, device)
    criterion = build_loss(arch, cfg)
    return model, criterion


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Novel Detection — Unified Training (DDP)")
    p.add_argument(
        "--arch",
        type=str,
        default="otdet",
        choices=["otdet", "wavedet", "scalenet", "toponet", "flownet"],
    )
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sync-bn", action="store_true", help="Use SyncBatchNorm for DDP")
    p.add_argument(
        "--find-unused",
        action="store_true",
        help="Set find_unused_parameters=True for DDP (needed for some arch variants)",
    )
    p.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to simulate larger batches",
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Run validation every N epochs (0 to disable)",
    )
    p.add_argument(
        "--logger",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "both", "none"],
        help="Experiment tracking backend",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="ionized-meteorite",
        help="WandB project name (used when --logger includes wandb)",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


REQUIRED_CONFIG_KEYS = {
    "model": ["num_classes", "feat_channels"],
    "data": ["img_size"],
    "train": [
        "epochs",
        "batch_size",
        "lr",
        "lr_min",
        "weight_decay",
        "warmup_epochs",
        "grad_clip",
    ],
    "loss": [
        "cls_weight",
        "reg_weight",
        "angle_weight",
        "iou_weight",
        "focal_alpha",
        "focal_gamma",
    ],
}

# Architecture-specific keys validated at warning level to catch typos
ARCH_SPECIFIC_KEYS = {
    "otdet": {
        "model": ["sinkhorn_iters", "sinkhorn_eps", "ot_cost_type"],
        "loss": ["ot_weight"],
    },
    "wavedet": {
        "model": ["num_wave_steps", "wave_dt"],
        "loss": ["conf_weight"],
    },
    "scalenet": {
        "model": ["num_scales", "sigma_min", "sigma_max"],
        "loss": ["conf_weight"],
    },
    "toponet": {
        "model": ["num_filtration_steps", "persistence_thresh"],
        "loss": ["conf_weight"],
    },
    "flownet": {
        "model": ["ode_steps", "ode_method"],
        "loss": ["conf_weight"],
    },
}


def validate_config(cfg: dict, arch: str = None):
    """Validate that all required config keys are present.

    Also warns about missing architecture-specific keys that will
    fall back to defaults, helping catch typos in config files.
    """
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in cfg:
            raise ValueError(f"Config missing required section: '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(
                    f"Config section '{section}' missing required key: '{key}'"
                )

    # Warn about missing arch-specific keys (fall back to factory defaults)
    if arch and arch in ARCH_SPECIFIC_KEYS:
        for section, keys in ARCH_SPECIFIC_KEYS[arch].items():
            if section not in cfg:
                continue
            for key in keys:
                if key not in cfg[section]:
                    warnings.warn(
                        f"[{arch}] Config section '{section}' missing optional "
                        f"key '{key}' — using factory default.",
                        stacklevel=2,
                    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train():
    args = parse_args()

    # DDP setup (noop for single-GPU)
    ddp_device, ddp_local_rank = setup_ddp()
    ddp_mode = is_dist()

    if args.device == "auto":
        device = ddp_device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    # Config
    if args.config is None:
        config_map = {
            "otdet": "configs/default.yaml",
            "wavedet": "configs/wavedet.yaml",
            "scalenet": "configs/scalenet.yaml",
            "toponet": "configs/toponet.yaml",
            "flownet": "configs/flownet.yaml",
        }
        args.config = config_map[args.arch]
    cfg = load_config(args.config)
    validate_config(cfg, arch=args.arch)

    epochs = args.epochs or cfg["train"]["epochs"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    lr = args.lr or cfg["train"]["lr"]
    save_dir = args.save_dir or f"runs/{args.arch}"
    img_size = cfg["data"]["img_size"]

    if is_main():
        world = get_world_size()
        print(
            f"[{args.arch.upper()}] Device: {device} | "
            f"DDP: {'ON' if ddp_mode else 'OFF'} | "
            f"GPUs: {world} | "
            f"Per-GPU batch: {batch_size} | "
            f"Effective batch: {batch_size * world}"
        )

    # Experiment tracking (rank-0 only, no-op if backend not installed)
    logger = ExperimentLogger(
        backend=args.logger if is_main() else "none",
        project=args.wandb_project,
        name=f"{args.arch}-{epochs}ep",
        config={
            "arch": args.arch,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "effective_lr": effective_lr,
            **cfg,
        },
        save_dir=os.path.join(save_dir, "tb_logs"),
        tags=[args.arch],
    )

    # Dataset
    train_dataset = YOLOOBBDataset(
        root=args.data,
        split="train",
        img_size=img_size,
        augment=cfg["data"]["augment"],
        hsv_h=cfg["data"]["hsv_h"],
        hsv_s=cfg["data"]["hsv_s"],
        hsv_v=cfg["data"]["hsv_v"],
        flip_lr=cfg["data"]["flip_lr"],
        flip_ud=cfg["data"]["flip_ud"],
    )

    # DDP sampler ensures each GPU gets unique data partition
    sampler = DistributedSampler(train_dataset, shuffle=True) if ddp_mode else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    if is_main():
        print(
            f"[{args.arch.upper()}] {len(train_dataset)} images, "
            f"{len(train_loader)} batches/epoch"
        )

    # Validation dataset (no augment, used for model selection)
    # Rank-0 only: no collective ops inside _run_validation.
    # For multi-GPU val aggregation, add DistributedSampler and all_reduce.
    val_loader = None
    if args.val_every > 0 and is_main():
        try:
            val_dataset = YOLOOBBDataset(
                root=args.data, split="val", img_size=img_size, augment=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
            print(f"[{args.arch.upper()}] Val: {len(val_dataset)} images")
        except FileNotFoundError:
            print(
                f"[{args.arch.upper()}] No val split found — using train loss for model selection"
            )

    # Model
    model, criterion = _get_model_and_loss(args.arch, cfg, img_size, device)

    # SyncBatchNorm for DDP
    if ddp_mode and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main():
            print(f"[{args.arch.upper()}] SyncBatchNorm enabled")

    # Wrap in DDP
    if ddp_mode:
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
            output_device=ddp_local_rank,
            find_unused_parameters=args.find_unused,
        )

    if is_main():
        raw = model.module if ddp_mode else model
        total_params = sum(p.numel() for p in raw.parameters())
        print(f"[{args.arch.upper()}] Parameters: {total_params:,}")

    # Optimizer + Scheduler
    # Scale LR and eta_min by world size (linear scaling rule)
    grad_clip = cfg["train"]["grad_clip"]
    warmup_epochs = cfg["train"]["warmup_epochs"]
    effective_lr = lr * get_world_size()
    effective_lr_min = cfg["train"]["lr_min"] * get_world_size()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=effective_lr, weight_decay=cfg["train"]["weight_decay"]
    )
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=effective_lr_min
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    # Resume
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = safe_load_checkpoint(args.resume, map_location=device)
        raw_model = model.module if ddp_mode else model
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        # Restore scheduler state (fallback: replay steps for old checkpoints)
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        if is_main():
            print(f"[{args.arch.upper()}] Resumed from epoch {start_epoch}")

    if is_main():
        os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()

        # DDP: set epoch for shuffling across workers
        if sampler is not None:
            sampler.set_epoch(epoch)

        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{args.arch}] Ep {epoch+1}/{epochs}",
            disable=not is_main(),
        )

        for step_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)

            preds = model(images)
            loss_dict = criterion(preds, targets, img_size=img_size)
            loss = loss_dict["total"] / args.accum_steps

            loss.backward()

            if (step_idx + 1) % args.accum_steps == 0 or (step_idx + 1) == len(
                train_loader
            ):
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss_dict["total"].item()
            n_batches += 1
            pbar.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        avg = epoch_loss / max(n_batches, 1)

        # Reduce loss across ranks for logging
        if ddp_mode:
            avg_tensor = torch.tensor([avg], device=device)
            dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
            avg = avg_tensor.item() / get_world_size()

        if is_main():
            print(f"[Epoch {epoch+1}] avg_loss={avg:.4f}")
            logger.log_scalars(
                {
                    "train/loss": avg,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1,
            )

            # Validation-based model selection
            selection_loss = avg
            if (
                val_loader is not None
                and args.val_every > 0
                and (epoch + 1) % args.val_every == 0
            ):
                val_avg = _run_validation(
                    model, criterion, val_loader, device, img_size, ddp_mode
                )
                print(f"  val_loss={val_avg:.4f}")
                selection_loss = val_avg
                logger.log_scalars({"val/loss": val_avg}, step=epoch + 1)

            # Build checkpoint after validation so best_loss is up-to-date
            is_best = selection_loss < best_loss
            if is_best:
                best_loss = selection_loss

            raw_model = model.module if ddp_mode else model
            ckpt = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "config": cfg,
                "arch": args.arch,
            }
            torch.save(ckpt, os.path.join(save_dir, "last.pt"))

            if is_best:
                torch.save(ckpt, os.path.join(save_dir, "best.pt"))
                print(f"  → Best model saved (loss={best_loss:.4f})")

    if is_main():
        print(f"\n[{args.arch.upper()}] Training complete. Best: {best_loss:.4f}")
        logger.finish()

    cleanup_ddp()


def _run_validation(model, criterion, val_loader, device, img_size, ddp_mode):
    """Run a single validation pass and return average loss."""
    raw_model = model.module if ddp_mode and hasattr(model, "module") else model
    raw_model.eval()
    val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            preds = raw_model(images)
            loss_dict = criterion(preds, targets, img_size=img_size)
            val_loss += loss_dict["total"].item()
            n_val += 1
    raw_model.train()
    return val_loss / max(n_val, 1)


if __name__ == "__main__":
    train()
