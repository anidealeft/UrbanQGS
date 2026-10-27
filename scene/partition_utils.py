import json
import os
from typing import Dict, List, Tuple

import numpy as np
from utils.graphics_utils import BasicPointCloud, getWorld2View2


def camera_center_from_cam_info(cam_info) -> np.ndarray:
    """Return world-space camera center from a CameraInfo-like object."""
    w2c = getWorld2View2(cam_info.R, cam_info.T)
    c2w = np.linalg.inv(w2c)
    return c2w[:3, 3].astype(np.float32)


def _augment_indices_to_min_count(centers_xy: np.ndarray,
                                  selected: np.ndarray,
                                  block_center_xy: np.ndarray,
                                  min_cameras: int) -> np.ndarray:
    """If a block has too few cameras, pull in the nearest remaining cameras."""
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size >= min_cameras:
        return selected

    d = np.linalg.norm(centers_xy - block_center_xy[None], axis=1)
    order = np.argsort(d)
    pool = list(selected.tolist())
    pool_set = set(pool)
    for idx in order.tolist():
        if idx in pool_set:
            continue
        pool.append(idx)
        pool_set.add(idx)
        if len(pool) >= min_cameras:
            break
    return np.asarray(sorted(pool), dtype=np.int64)


def build_regular_grid_blocks(cam_infos: List,
                              grid_x: int,
                              grid_y: int,
                              overlap_ratio: float = 0.10,
                              min_cameras: int = 8) -> List[Dict]:
    """
    Build a regular XY camera grid for scene-level partitioned training.

    Returns a list of block metadata dictionaries containing both exact bounds and
    overlap-expanded bounds.
    """
    if len(cam_infos) == 0:
        return []

    centers = np.stack([camera_center_from_cam_info(c) for c in cam_infos], axis=0)
    xy = centers[:, :2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)

    dx = max((xmax - xmin) / max(grid_x, 1), 1e-6)
    dy = max((ymax - ymin) / max(grid_y, 1), 1e-6)
    mx = dx * max(overlap_ratio, 0.0)
    my = dy * max(overlap_ratio, 0.0)

    blocks: List[Dict] = []
    for by in range(max(grid_y, 1)):
        for bx in range(max(grid_x, 1)):
            x0 = xmin + bx * dx
            x1 = xmin + (bx + 1) * dx
            y0 = ymin + by * dy
            y1 = ymin + (by + 1) * dy

            exact_mask = (
                (xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
                (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            )
            overlap_mask = (
                (xy[:, 0] >= x0 - mx) & (xy[:, 0] <= x1 + mx) &
                (xy[:, 1] >= y0 - my) & (xy[:, 1] <= y1 + my)
            )

            selected = np.where(overlap_mask)[0]
            block_center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            selected = _augment_indices_to_min_count(xy, selected, block_center, min_cameras)

            blocks.append({
                "block_id": len(blocks),
                "grid_coord": [int(bx), int(by)],
                "bbox_xy_exact": [float(x0), float(y0), float(x1), float(y1)],
                "bbox_xy_overlap": [float(x0 - mx), float(y0 - my), float(x1 + mx), float(y1 + my)],
                "camera_indices": selected.tolist(),
                "camera_indices_exact": np.where(exact_mask)[0].tolist(),
            })
    return blocks


def _filter_test_cameras_by_bbox(test_cam_infos: List, bbox_xy_exact: List[float]) -> List:
    if len(test_cam_infos) == 0:
        return []
    x0, y0, x1, y1 = bbox_xy_exact
    selected = []
    for cam in test_cam_infos:
        center = camera_center_from_cam_info(cam)
        if x0 <= center[0] <= x1 and y0 <= center[1] <= y1:
            selected.append(cam)
    return selected



def filter_scene_to_block(train_cam_infos: List,
                          test_cam_infos: List,
                          pcd,
                          block_meta: Dict,
                          point_pad: float = 0.15) -> Tuple[List, List, BasicPointCloud, Dict]:
    """
    Filter cameras and initial point cloud to a block.

    Train views use overlap-expanded bounds. Test views use exact bounds to avoid duplicated evaluation.
    Point cloud is filtered in XY and left unconstrained in Z.
    """
    train_ids = [i for i in block_meta["camera_indices"] if i < len(train_cam_infos)]
    block_train = [train_cam_infos[i] for i in train_ids]
    block_test = _filter_test_cameras_by_bbox(test_cam_infos, block_meta["bbox_xy_exact"])

    block_pcd = pcd
    if pcd is not None:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals = None if getattr(pcd, "normals", None) is None else np.asarray(pcd.normals)

        x0, y0, x1, y1 = block_meta["bbox_xy_overlap"]
        dx = max(x1 - x0, 1e-6)
        dy = max(y1 - y0, 1e-6)
        px = dx * max(point_pad, 0.0)
        py = dy * max(point_pad, 0.0)

        mask = (
            (points[:, 0] >= x0 - px) & (points[:, 0] <= x1 + px) &
            (points[:, 1] >= y0 - py) & (points[:, 1] <= y1 + py)
        )
        if mask.any():
            block_pcd = BasicPointCloud(
                points=points[mask],
                colors=colors[mask],
                normals=None if normals is None else normals[mask],
            )

    meta = dict(block_meta)
    meta.update({
        "num_train_cameras": len(block_train),
        "num_test_cameras": len(block_test),
        "num_points": 0 if block_pcd is None else int(np.asarray(block_pcd.points).shape[0]),
    })
    return block_train, block_test, block_pcd, meta


def write_block_manifest(path: str, meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
