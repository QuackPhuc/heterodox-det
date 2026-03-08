"""Detection model architectures.

Public API — all architectures importable from ``models``:

    from models import OTDet, WaveDetNet, ScaleNet, TopoNet, FlowNet
"""

from .otdet import OTDet
from .wavedet import WaveDetNet
from .scalenet import ScaleNet
from .toponet import TopoNet
from .flownet import FlowNet

__all__ = ["OTDet", "WaveDetNet", "ScaleNet", "TopoNet", "FlowNet"]
