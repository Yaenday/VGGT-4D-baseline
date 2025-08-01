#!/usr/bin/env python3
"""
monst3r_glb_to_colmap.py
从 MonST3R scene.glb 提取点云 → COLMAP
自动下采样至 ≤ 100000 点
"""

import numpy as np
import trimesh
from pathlib import Path
import sys

from dust3r.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
import cv2


def load_from_glb(root: Path, max_points: int = 100000):
    glb_path = root / 'scene.glb'
    mesh = trimesh.load(glb_path)

    if isinstance(mesh, trimesh.Scene):
        pts = []
        colors = []
        for node in mesh.geometry.values():
            if isinstance(node, trimesh.PointCloud):
                pts.append(node.vertices)
                colors.append(node.colors[:, :3])
            else:
                pts.append(node.vertices)
                colors.append(np.tile([255, 255, 255], (len(node.vertices), 1)))
        points3d = np.concatenate(pts).astype(np.float32)
        points_rgb = np.concatenate(colors).astype(np.uint8)
    else:
        if isinstance(mesh, trimesh.PointCloud):
            points3d = mesh.vertices.astype(np.float32)
            points_rgb = mesh.colors[:, :3].astype(np.uint8)
        else:
            raise ValueError("GLB 中未找到 PointCloud")

    # 下采样
    if len(points3d) > max_points:
        idx = np.random.choice(len(points3d), max_points, replace=False)
        points3d = points3d[idx]
        points_rgb = points_rgb[idx]

    # 2. 内参
    intrin_path = root / 'pred_intrinsics.txt'
    intrinsics = [np.array(list(map(float, l.split()))).reshape(3, 3)
                  for l in intrin_path.read_text().strip().splitlines()]
    intrinsics = np.stack(intrinsics)

    # 3. 外参
    traj_path = root / 'pred_traj.txt'
    extrinsics = []
    from scipy.spatial.transform import Rotation as R
    for line in traj_path.read_text().strip().splitlines():
        ts, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
        Rmat = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz])
        extrinsics.append(np.hstack([Rmat, t.reshape(-1, 1)]))
    extrinsics = np.stack(extrinsics)

    # 4. 图像尺寸
    png_paths = sorted(root.glob('frame_*.png'))
    H, W = cv2.imread(str(png_paths[0])).shape[:2]
    image_size = np.array([W, H])

    # 5. 构造 dummy points_xyf（每点随机分配到某帧）
    N_frames = len(png_paths)
    frame_ids = np.random.randint(0, N_frames, size=len(points3d))
    u = np.random.randint(0, W, size=len(points3d))
    v = np.random.randint(0, H, size=len(points3d))
    points_xyf = np.stack([u, v, frame_ids], axis=1)

    return points3d, points_rgb, points_xyf, extrinsics, intrinsics, image_size


def main(root_dir: str):
    root = Path(root_dir)
    out_dir = root / 'sparse' / '0'
    out_dir.mkdir(parents=True, exist_ok=True)

    pts3d, rgb, xyf, ext, K, img_size = load_from_glb(root, max_points=100000)
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points3d=pts3d,
        points_xyf=xyf,
        points_rgb=rgb,
        extrinsics=ext,
        intrinsics=K,
        image_size=img_size,
        shared_camera=False,
        camera_type='PINHOLE'
    )
    reconstruction.write(str(out_dir))
    trimesh.PointCloud(pts3d, colors=rgb).export(str(out_dir / 'points.ply'))
    print(f"COLMAP sparse reconstruction saved to {out_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('monst3r_output_dir', type=str)
    args = parser.parse_args()
    main(args.monst3r_output_dir)