#!/usr/bin/env python3
"""
MonST3R → COLMAP 一步到位
用法：
python demo_colmap.py --input_dir demo_data/lady-running \
                      --output_dir demo_tmp/lady-running \
                      --seq_name lady-running
生成目录结构：
<output_dir>/sparse/0/
├── cameras.bin
├── images.bin
├── points3D.bin
└── points.ply
"""

import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

# ---- 复用 VGGT 的转换函数 ----
from dust3r.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

# -------------------------------------------------
# 1. 参数解析
# -------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True,
                   help='文件夹或单视频，含连续 JPEG')
    p.add_argument('--output_dir', default='demo_tmp')
    p.add_argument('--seq_name', default='NULL')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--weights', default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    p.add_argument('--device', default='cuda')
    p.add_argument('--niter', type=int, default=300)
    p.add_argument('--min_conf_thr', type=float, default=1.1)
    p.add_argument('--shared_focal', action='store_true', default=True)
    p.add_argument('--scenegraph_type', default='swinstride-5-noncyclic')
    p.add_argument('--batch_size', type=int, default=16)
    return p.parse_args()


# -------------------------------------------------
# 2. 主流程
# -------------------------------------------------
@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)

    # ---- 2.1 加载模型 ----
    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(device)
    model.eval()

    # ---- 2.2 加载图像 ----
    filelist = sorted([str(p) for p in Path(args.input_dir).glob('*.jpg')]) + \
            sorted([str(p) for p in Path(args.input_dir).glob('*.png')])
    if not filelist:
        raise FileNotFoundError('No jpg/png found in input_dir')
    imgs = load_images(filelist, size=args.image_size, verbose=False)

    # ---- 2.3 推理 + 全局优化 ----
    pairs = make_pairs(imgs, scene_graph=args.scenegraph_type,
                       prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=args.batch_size)
    mode = GlobalAlignerMode.PointCloudOptimizer
    scene = global_aligner(output, device=device, mode=mode, shared_focal=args.shared_focal,
                           num_total_iter=args.niter, verbose=False)
    scene.compute_global_alignment(init='mst', niter=args.niter)

    scene = global_aligner(output, device=device, mode=mode, shared_focal=args.shared_focal,
                       num_total_iter=args.niter, verbose=False)

    scene.compute_global_alignment(init='mst', niter=args.niter)

    # -------------------------------------------------
    # 3. 提取 COLMAP 所需变量
    # -------------------------------------------------
    H, W = imgs[0]['img'].shape[:2]
    image_size = np.array([W, H])

    # 3.1 内参 (所有帧共享)
    fx = fy = scene.get_focals()[0].item()
    cx, cy = W / 2.0, H / 2.0          # MonST3R 中心化主点
    intrinsics = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
    intrinsics = np.stack([intrinsics] * len(imgs))   # (N,3,3)

    # 3.2 外参 (TUM → COLMAP)
    poses = scene.get_im_poses().cpu().numpy()          # (N,4,4) c2w
    extrinsics = []
    for c2w in poses:
        w2c = np.linalg.inv(c2w)
        extrinsics.append(w2c[:3, :])
    extrinsics = np.stack(extrinsics)                   # (N,3,4)

    # 3.3 3D 点云 + 颜色 + 像素坐标
    pts3d = to_numpy(scene.get_pts3d())                 # list of (H,W,3)
    conf  = to_numpy(scene.get_conf())                  # list of (H,W)
    rgbs  = [to_numpy(im['img']) for im in imgs]        # list of (H,W,3)

    points3d_list, points_rgb_list, points_xyf_list = [], [], []
    for fidx, (pts, mask, rgb) in enumerate(zip(pts3d, conf, rgbs)):
        mask = mask > args.min_conf_thr
        pts_valid  = pts[mask]
        rgb_valid  = rgb[mask]
        yv, xv = np.where(mask)
        xyf = np.stack([xv, yv, np.ones_like(xv) * fidx], axis=1)
        points3d_list.append(pts_valid)
        points_rgb_list.append(rgb_valid)
        points_xyf_list.append(xyf)

    points3d = np.concatenate(points3d_list)
    points_rgb = np.concatenate(points_rgb_list)
    points_xyf = np.concatenate(points_xyf_list)

    # -------------------------------------------------
    # 4. 写入 COLMAP
    # -------------------------------------------------
    out_root = Path(args.output_dir) / args.seq_name / 'sparse' / '0'
    out_root.mkdir(parents=True, exist_ok=True)

    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points3d=points3d,
        points_xyf=points_xyf,
        points_rgb=points_rgb,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_size=image_size,
        shared_camera=args.shared_focal,
        camera_type='PINHOLE'
    )
    reconstruction.write(str(out_root))

    # 额外导出一份 ply 便于查看
    ply_path = out_root / 'points.ply'
    import trimesh
    trimesh.PointCloud(points3d, colors=points_rgb).export(str(ply_path))
    print(f"COLMAP format saved to {out_root}")


if __name__ == '__main__':
    main()