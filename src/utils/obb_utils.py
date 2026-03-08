"""OBB (Oriented Bounding Box) geometry utilities.

All angles are in radians. Coordinates are absolute (not normalized)
unless stated otherwise.
"""

import math
import numpy as np
import torch

try:
    from shapely.geometry import Polygon as ShapelyPolygon

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def poly_to_obb(poly: np.ndarray) -> np.ndarray:
    """Convert 4-point polygon to oriented bounding box.

    Args:
        poly: (N, 8) array of x1,y1,...,x4,y4  OR  (8,)

    Returns:
        obb: (N, 5) or (5,) of cx, cy, w, h, angle (radians, [-pi/2, pi/2))
    """
    single = poly.ndim == 1
    if single:
        poly = poly[np.newaxis, :]

    pts = poly.reshape(-1, 4, 2).astype(np.float32)

    obbs = []
    for p in pts:
        # Use OpenCV-free minimum-area rectangle via covariance
        cx, cy = p.mean(axis=0)

        # Compute edges and pick the longest to define orientation
        edges = np.diff(np.vstack([p, p[0:1]]), axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        longest_idx = np.argmax(lengths)
        edge = edges[longest_idx]
        angle = math.atan2(edge[1], edge[0])

        # Normalize angle to [-pi/2, pi/2)
        while angle >= math.pi / 2:
            angle -= math.pi
        while angle < -math.pi / 2:
            angle += math.pi

        # Rotate points to axis-aligned frame
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        p_rot = (p - np.array([cx, cy])) @ rot.T

        w = p_rot[:, 0].max() - p_rot[:, 0].min()
        h = p_rot[:, 1].max() - p_rot[:, 1].min()

        obbs.append([cx, cy, w, h, angle])

    result = np.array(obbs, dtype=np.float32)
    return result[0] if single else result


def obb_to_poly(obb: np.ndarray) -> np.ndarray:
    """Convert OBB (cx, cy, w, h, angle) to 4-point polygon.

    Args:
        obb: (N, 5) or (5,)

    Returns:
        poly: (N, 8) or (8,) of x1,y1,...,x4,y4
    """
    single = obb.ndim == 1
    if single:
        obb = obb[np.newaxis, :]

    cx, cy, w, h, angle = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]

    # Half dimensions
    dx = w / 2
    dy = h / 2

    # Corner offsets in local frame (CW order in image coords: y-down)
    corners_x = np.stack([-dx, dx, dx, -dx], axis=1)
    corners_y = np.stack([dy, dy, -dy, -dy], axis=1)

    cos_a = np.cos(angle)[:, None]
    sin_a = np.sin(angle)[:, None]

    # Rotate corners
    rx = corners_x * cos_a - corners_y * sin_a + cx[:, None]
    ry = corners_x * sin_a + corners_y * cos_a + cy[:, None]

    polys = np.stack([rx, ry], axis=2).reshape(-1, 8)
    return polys[0] if single else polys


def _polygon_iou_single(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Compute IoU between two polygons (each 8 values)."""
    if HAS_SHAPELY:
        p1 = ShapelyPolygon(poly1.reshape(4, 2))
        p2 = ShapelyPolygon(poly2.reshape(4, 2))
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / union if union > 0 else 0.0
    else:
        # Approximate with axis-aligned IoU
        obb1 = poly_to_obb(poly1)
        obb2 = poly_to_obb(poly2)
        x1 = max(obb1[0] - obb1[2] / 2, obb2[0] - obb2[2] / 2)
        y1 = max(obb1[1] - obb1[3] / 2, obb2[1] - obb2[3] / 2)
        x2 = min(obb1[0] + obb1[2] / 2, obb2[0] + obb2[2] / 2)
        y2 = min(obb1[1] + obb1[3] / 2, obb2[1] + obb2[3] / 2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = obb1[2] * obb1[3]
        area2 = obb2[2] * obb2[3]
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0


def _gaussian_obb_iou(obbs1: np.ndarray, obbs2: np.ndarray) -> np.ndarray:
    """Vectorized approximate OBB IoU via Gaussian Bhattacharyya distance.

    Uses the same mathematical proxy as obb_iou_tensor (the differentiable
    version), ported to numpy for fast CPU evaluation and NMS.

    Args:
        obbs1: (M, 5) — cx, cy, w, h, angle
        obbs2: (N, 5)

    Returns:
        iou_matrix: (M, N)
    """

    def _to_gaussian(obb):
        cx, cy, w, h, a = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        var_w = (w**2) / 12.0
        var_h = (h**2) / 12.0
        s11 = var_w * cos_a**2 + var_h * sin_a**2
        s22 = var_w * sin_a**2 + var_h * cos_a**2
        s12 = (var_w - var_h) * cos_a * sin_a
        return cx, cy, s11, s22, s12

    cx1, cy1, s11_1, s22_1, s12_1 = _to_gaussian(obbs1)  # each (M,)
    cx2, cy2, s11_2, s22_2, s12_2 = _to_gaussian(obbs2)  # each (N,)

    # Broadcast to (M, N)
    dcx = cx1[:, None] - cx2[None, :]
    dcy = cy1[:, None] - cy2[None, :]

    # Average covariance entries
    a11 = (s11_1[:, None] + s11_2[None, :]) / 2.0 + 1e-6
    a22 = (s22_1[:, None] + s22_2[None, :]) / 2.0 + 1e-6
    a12 = (s12_1[:, None] + s12_2[None, :]) / 2.0

    # Determinant and inverse of 2x2 average covariance
    det_a = a11 * a22 - a12 * a12
    det_a = np.clip(det_a, 1e-8, None)
    inv11 = a22 / det_a
    inv22 = a11 / det_a
    inv12 = -a12 / det_a

    # Mahalanobis distance
    maha = dcx * dcx * inv11 + 2 * dcx * dcy * inv12 + dcy * dcy * inv22

    # Log det ratio
    det1 = s11_1 * s22_1 - s12_1 * s12_1
    det2 = s11_2 * s22_2 - s12_2 * s12_2
    log_det_ratio = np.log(np.clip(det_a, 1e-8, None)) - 0.5 * (
        np.log(np.clip(det1[:, None], 1e-8, None))
        + np.log(np.clip(det2[None, :], 1e-8, None))
    )

    bhatt = 0.125 * maha + 0.5 * log_det_ratio
    iou = np.clip(np.exp(-bhatt), 0.0, 1.0)
    return iou.astype(np.float32)


def obb_iou(obbs1: np.ndarray, obbs2: np.ndarray, exact: bool = False) -> np.ndarray:
    """Compute pairwise IoU between two sets of OBBs.

    Uses a fast vectorized Gaussian proxy by default. Pass exact=True
    for Shapely polygon intersection (slower but geometrically exact).

    Args:
        obbs1: (M, 5) — cx, cy, w, h, angle
        obbs2: (N, 5)
        exact: use Shapely polygon intersection instead of Gaussian proxy

    Returns:
        iou_matrix: (M, N)
    """
    if not exact:
        return _gaussian_obb_iou(obbs1, obbs2)

    polys1 = obb_to_poly(obbs1)
    polys2 = obb_to_poly(obbs2)
    M, N = len(obbs1), len(obbs2)
    iou = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            iou[i, j] = _polygon_iou_single(polys1[i], polys2[j])
    return iou


def obb_nms(
    obbs: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5
) -> np.ndarray:
    """Oriented bounding box NMS.

    Args:
        obbs: (N, 5) — cx, cy, w, h, angle
        scores: (N,)
        iou_thresh: IoU threshold for suppression

    Returns:
        keep: indices of kept detections
    """
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        remaining = order[1:]
        ious = obb_iou(obbs[i : i + 1], obbs[remaining]).flatten()
        mask = ious < iou_thresh
        order = remaining[mask]

    return np.array(keep, dtype=np.int64)


def obb_iou_tensor(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable approximate OBB IoU using Gaussian assumption.

    Uses the Gaussian Wasserstein distance as a proxy for OBB IoU,
    which is differentiable and suitable for loss computation.

    Args:
        pred: (N, 5) — cx, cy, w, h, angle
        target: (N, 5)

    Returns:
        iou: (N,) approximate IoU values
    """

    # Convert OBB to 2D Gaussian (center, covariance)
    def obb_to_gaussian(obb):
        cx, cy = obb[:, 0], obb[:, 1]
        w, h, a = obb[:, 2], obb[:, 3], obb[:, 4]
        cos_a = torch.cos(a)
        sin_a = torch.sin(a)
        # Covariance: R @ diag(w²/12, h²/12) @ R^T
        # (uniform distribution over rectangle has variance dim²/12)
        var_w = (w**2) / 12.0
        var_h = (h**2) / 12.0
        s11 = var_w * cos_a**2 + var_h * sin_a**2
        s22 = var_w * sin_a**2 + var_h * cos_a**2
        s12 = (var_w - var_h) * cos_a * sin_a
        mu = torch.stack([cx, cy], dim=-1)
        sigma = torch.stack([s11, s12, s12, s22], dim=-1).reshape(-1, 2, 2)
        return mu, sigma

    mu1, s1 = obb_to_gaussian(pred)
    mu2, s2 = obb_to_gaussian(target)

    # Bhattacharyya-like distance as IoU proxy
    s_avg = (s1 + s2) / 2.0
    diff = (mu1 - mu2).unsqueeze(-1)

    # Regularize for numerical stability
    s_avg = s_avg + 1e-6 * torch.eye(2, device=pred.device).unsqueeze(0)

    det_s_avg = s_avg[:, 0, 0] * s_avg[:, 1, 1] - s_avg[:, 0, 1] * s_avg[:, 1, 0]
    det_s1 = s1[:, 0, 0] * s1[:, 1, 1] - s1[:, 0, 1] * s1[:, 1, 0]
    det_s2 = s2[:, 0, 0] * s2[:, 1, 1] - s2[:, 0, 1] * s2[:, 1, 0]

    # Inverse of s_avg (2x2)
    inv_s = torch.zeros_like(s_avg)
    inv_s[:, 0, 0] = s_avg[:, 1, 1]
    inv_s[:, 1, 1] = s_avg[:, 0, 0]
    inv_s[:, 0, 1] = -s_avg[:, 0, 1]
    inv_s[:, 1, 0] = -s_avg[:, 1, 0]
    inv_s = inv_s / det_s_avg.clamp(min=1e-8).unsqueeze(-1).unsqueeze(-1)

    # Mahalanobis distance
    maha = (
        torch.matmul(torch.matmul(diff.transpose(-1, -2), inv_s), diff)
        .squeeze(-1)
        .squeeze(-1)
    )
    # Log determinant ratio
    log_det_ratio = torch.log(det_s_avg.clamp(min=1e-8)) - 0.5 * (
        torch.log(det_s1.clamp(min=1e-8)) + torch.log(det_s2.clamp(min=1e-8))
    )

    # Bhattacharyya distance
    bhatt = 0.125 * maha + 0.5 * log_det_ratio
    # Convert to IoU-like score in [0, 1]
    iou = torch.exp(-bhatt).clamp(0.0, 1.0)
    return iou
