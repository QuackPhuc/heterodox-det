"""Shared model and loss construction utilities.

Centralizes architecture instantiation so train.py, test.py, and
dry_run.py all share one source of truth.
"""

import torch
import torch.nn as nn

from models import OTDet, WaveDetNet, ScaleNet, TopoNet, FlowNet
from losses import OTDetLoss, PeakDetLoss


# Model-to-loss mapping: OTDet uses OTDetLoss, all others use PeakDetLoss
_OT_ARCHS = {"otdet"}


def build_model(arch: str, cfg: dict, img_size: int, device: torch.device) -> nn.Module:
    """Construct a detection model from architecture name and config.

    Args:
        arch: one of 'otdet', 'wavedet', 'scalenet', 'toponet', 'flownet'
        cfg: full config dict (must contain 'model' section)
        img_size: input image size
        device: target device

    Returns:
        Instantiated model on the given device.
    """
    nc = cfg["model"]["num_classes"]
    fc = cfg["model"]["feat_channels"]
    np_ = cfg["model"].get("num_proposals", cfg["model"].get("num_slots", 100))

    if arch == "otdet":
        return OTDet(
            num_classes=nc,
            num_slots=np_,
            feat_channels=fc,
            pretrained_backbone=cfg["model"].get("pretrained", True),
            sinkhorn_iters=cfg["model"].get("sinkhorn_iters", 20),
            sinkhorn_eps=cfg["model"].get("sinkhorn_eps", 0.05),
            ot_cost_type=cfg["model"].get("ot_cost_type", "l2"),
            img_size=img_size,
        ).to(device)
    elif arch == "wavedet":
        return WaveDetNet(
            num_classes=nc,
            feat_channels=fc,
            num_proposals=np_,
            num_wave_steps=cfg["model"].get("num_wave_steps", 8),
            wave_dt=cfg["model"].get("wave_dt", 0.3),
            img_size=img_size,
        ).to(device)
    elif arch == "scalenet":
        return ScaleNet(
            num_classes=nc,
            feat_channels=fc,
            num_proposals=np_,
            num_scales=cfg["model"].get("num_scales", 8),
            sigma_range=(
                cfg["model"].get("sigma_min", 0.5),
                cfg["model"].get("sigma_max", 8.0),
            ),
            img_size=img_size,
        ).to(device)
    elif arch == "toponet":
        return TopoNet(
            num_classes=nc,
            feat_channels=fc,
            num_proposals=np_,
            num_filtration_steps=cfg["model"].get("num_filtration_steps", 16),
            persistence_thresh=cfg["model"].get("persistence_thresh", 0.1),
            img_size=img_size,
        ).to(device)
    elif arch == "flownet":
        return FlowNet(
            num_classes=nc,
            feat_channels=fc,
            num_proposals=np_,
            ode_steps=cfg["model"].get("ode_steps", 8),
            ode_method=cfg["model"].get("ode_method", "euler"),
            img_size=img_size,
        ).to(device)
    raise ValueError(f"Unknown architecture: {arch}")


def build_loss(arch: str, cfg: dict) -> nn.Module:
    """Construct the loss function for the given architecture.

    Args:
        arch: architecture name
        cfg: full config dict (must contain 'model' and 'loss' sections)

    Returns:
        Loss module instance.
    """
    nc = cfg["model"]["num_classes"]
    lc = cfg["loss"]

    if arch in _OT_ARCHS:
        return OTDetLoss(
            num_classes=nc,
            cls_weight=lc["cls_weight"],
            reg_weight=lc["reg_weight"],
            ot_weight=lc.get("ot_weight", 1.0),
            angle_weight=lc["angle_weight"],
            iou_weight=lc["iou_weight"],
            focal_alpha=lc["focal_alpha"],
            focal_gamma=lc["focal_gamma"],
        )
    return PeakDetLoss(
        num_classes=nc,
        cls_weight=lc["cls_weight"],
        reg_weight=lc["reg_weight"],
        angle_weight=lc["angle_weight"],
        iou_weight=lc["iou_weight"],
        conf_weight=lc.get("conf_weight", 1.0),
        focal_alpha=lc["focal_alpha"],
        focal_gamma=lc["focal_gamma"],
    )
