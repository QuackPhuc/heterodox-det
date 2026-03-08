# Novel Detection Architectures

5 fundamentally new object detection architectures, each based on a different mathematical framework — **none are CNN/Transformer hybrids**.

| Architecture | Core Principle | Params | Config |
| --- | --- | --- | --- |
| **OTDet** | Sinkhorn Optimal Transport | 12.0M | `configs/otdet.yaml` |
| **WaveDetNet** | Wave Equation Resonance | 1.2M | `configs/wavedet.yaml` |
| **ScaleNet** | Continuous Scale-Space (SIREN) | 231K | `configs/scalenet.yaml` |
| **TopoNet** | Topological Persistence | 429K | `configs/toponet.yaml` |
| **FlowNet** | Neural ODE Attractors | 307K | `configs/flownet.yaml` |

## Getting Started

```bash
pip install -r requirements.txt
python scripts/dry_run.py          # Smoke test all 5 architectures
python -m pytest tests/ -q         # Run test suite
```

## Training

```bash
python train.py --arch <arch> --data <path> [options]
```

| Arg | Default | Description |
| --- | --- | --- |
| `--arch` | `otdet` | `otdet` · `wavedet` · `scalenet` · `toponet` · `flownet` |
| `--data` | *required* | Dataset root (YOLO OBB v5 format) |
| `--config` | per-arch | YAML config override |
| `--epochs` | config | Training epochs |
| `--batch-size` | config | Per-GPU batch size |
| `--lr` | config | Base learning rate (auto-scaled for DDP) |
| `--accum-steps` | `1` | Gradient accumulation (effective batch = batch × accum × GPUs) |
| `--resume` | — | Resume from `.pt` checkpoint |
| `--save-dir` | `runs/<arch>` | Output directory for `last.pt` / `best.pt` |
| `--val-every` | `5` | Validation interval in epochs (`0` = disable) |
| `--device` | `auto` | `auto` · `cpu` · `cuda` · `cuda:N` |
| `--num-workers` | `4` | DataLoader workers |
| `--sync-bn` | off | SyncBatchNorm for DDP |
| `--find-unused` | off | DDP `find_unused_parameters` |
| `--logger` | `none` | `wandb` · `tensorboard` · `both` · `none` |
| `--wandb-project` | `ionized-meteorite` | WandB project name |

```bash
# Single GPU
python train.py --arch otdet --data ./data/dota

# Multi-GPU DDP
torchrun --nproc_per_node=4 train.py --arch wavedet --data ./data/dota --sync-bn

# Custom hyperparams + gradient accumulation + logging
python train.py --arch scalenet --data ./data/dota \
    --lr 5e-4 --batch-size 4 --accum-steps 4 --logger wandb
```

## Evaluation

```bash
python test.py --data <path> --weights <checkpoint> [options]
```

| Arg | Default | Description |
| --- | --- | --- |
| `--data` | *required* | Dataset root |
| `--weights` | *required* | Checkpoint `.pt` path |
| `--arch` | auto | Auto-detected from checkpoint |
| `--split` | `val` | `train` · `val` · `test` |
| `--batch-size` | `8` | Batch size |
| `--conf` | `0.25` | Confidence threshold |
| `--nms` | `0.45` | OBB NMS IoU threshold |
| `--device` | `auto` | Device |

```bash
python test.py --data ./data/dota --weights runs/otdet/best.pt
python test.py --data ./data/dota --weights runs/wavedet/best.pt --conf 0.5 --split test
```

## Experiment Tracking

Both backends are **optional** — training works identically without them.

| Backend | Install | Setup | Logs |
| --- | --- | --- | --- |
| **TensorBoard** | `pip install tensorboard` | None needed | Local: `runs/<arch>/tb_logs/` |
| **WandB** | `pip install wandb` | `wandb login` ([get API key](https://wandb.ai/authorize)) | Cloud: [wandb.ai](https://wandb.ai) |

```bash
# TensorBoard
python train.py --arch otdet --data ./data/dota --logger tensorboard
tensorboard --logdir runs/otdet/tb_logs    # http://localhost:6006

# WandB
wandb login                                 # one-time — paste API key
python train.py --arch otdet --data ./data/dota --logger wandb
```

Logged: `train/loss`, `train/lr` (every epoch), `val/loss` (every `--val-every` epochs).

## Dataset Format

YOLO OBB v5: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (normalized [0,1])

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Project Structure

```
├── train.py                 # Unified training (supports DDP)
├── test.py                  # Evaluation
├── configs/                 # YAML configs per architecture
├── src/
│   ├── models/              # 5 architectures + shared InferenceMixin
│   ├── losses/              # OTDetLoss + PeakDetLoss (shared)
│   ├── data/dataset.py      # YOLO OBB v5 loader + augmentation
│   └── utils/               # factory, obb_utils, metrics, inference, checkpoint, logger
├── tests/                   # 91 tests (models, losses, metrics, OBB, integration)
├── scripts/dry_run.py       # Smoke test for all architectures
└── docs/                    # Research notes + per-model architecture docs
```

## Documentation

Detailed architecture docs (theory, math, analysis per model) are in `docs/` — see `docs/README.md` for the full index.

## Theory

Each architecture replaces a fundamental assumption of conventional detection:

| Architecture | Replaces | With |
| --- | --- | --- |
| **OTDet** | Discriminative classification | Kantorovich optimal transport |
| **WaveDetNet** | Feedforward layers | Wave equation physics |
| **ScaleNet** | Discrete feature pyramid | Continuous scale-space (SIREN) |
| **TopoNet** | Bounding box representation | Persistent topological features |
| **FlowNet** | Layer stacking | Contractive ODE dynamics |
