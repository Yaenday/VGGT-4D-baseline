#!/usr/bin/env python3
"""
monst3r_to_colmap_batch.py
Batch convert MonST3R outputs to COLMAP format for all datasets
"""

import numpy as np
import trimesh
from pathlib import Path
import sys
import argparse

# Add the dust3r path to sys.path
sys.path.append(str(Path(__file__).parent))

from dust3r.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
import cv2


def load_from_glb(root: Path, max_points: int = 100000):
    """
    Load data from MonST3R output directory
    """
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

    # Downsample if needed
    if len(points3d) > max_points:
        idx = np.random.choice(len(points3d), max_points, replace=False)
        points3d = points3d[idx]
        points_rgb = points_rgb[idx]

    # Load intrinsics
    intrin_path = root / 'pred_intrinsics.txt'
    intrinsics = [np.array(list(map(float, l.split()))).reshape(3, 3)
                  for l in intrin_path.read_text().strip().splitlines()]
    intrinsics = np.stack(intrinsics)

    # Load extrinsics
    traj_path = root / 'pred_traj.txt'
    extrinsics = []
    from scipy.spatial.transform import Rotation as R
    for line in traj_path.read_text().strip().splitlines():
        ts, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
        Rmat = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz])
        extrinsics.append(np.hstack([Rmat, t.reshape(-1, 1)]))
    extrinsics = np.stack(extrinsics)

    # Get image dimensions
    png_paths = sorted(root.glob('frame_*.png'))
    if not png_paths:
        raise FileNotFoundError("No frame PNG files found in directory")
    
    H, W = cv2.imread(str(png_paths[0])).shape[:2]
    image_size = np.array([W, H])

    # Construct dummy points_xyf (randomly assign each point to a frame)
    N_frames = len(png_paths)
    frame_ids = np.random.randint(0, N_frames, size=len(points3d))
    u = np.random.randint(0, W, size=len(points3d))
    v = np.random.randint(0, H, size=len(points3d))
    points_xyf = np.stack([u, v, frame_ids], axis=1)

    return points3d, points_rgb, points_xyf, extrinsics, intrinsics, image_size


def process_single_dataset(dataset_path: Path):
    """
    Process a single dataset directory
    """
    print(f"Processing dataset: {dataset_path.name}")
    
    # Find the output directory with the same name as the dataset
    output_dir = dataset_path / "output" 
    
    if not output_dir.exists():
        print(f"  Warning: Output directory not found: {output_dir}")
        return False
    
    # Create sparse output directory
    sparse_dir = dataset_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data from GLB
        pts3d, rgb, xyf, ext, K, img_size = load_from_glb(output_dir, max_points=500000)
        
        # Convert to COLMAP format
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
        
        # Save reconstruction
        reconstruction.write(str(sparse_dir))
        
        # Save PLY file
        trimesh.PointCloud(pts3d, colors=rgb).export(str(sparse_dir / 'points.ply'))
        
        print(f"  Successfully saved COLMAP reconstruction to {sparse_dir}")
        return True
        
    except Exception as e:
        print(f"  Error processing {dataset_path.name}: {e}")
        return False


def process_all_datasets(base_path: Path):
    """
    Process all datasets in the base directory
    """
    print(f"Processing all datasets in: {base_path}")
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    success_count = 0
    total_count = 0
    
    # Iterate through dataset directories
    for dataset_dir in base_path.iterdir():
        for subdir in dataset_dir.iterdir():
            if subdir.is_dir():
                total_count += 1
                if process_single_dataset(subdir):
                    success_count += 1
    
    print(f"\nProcessing complete. Success: {success_count}/{total_count} datasets")


def main():
    parser = argparse.ArgumentParser(description="Batch convert MonST3R outputs to COLMAP format")
    parser.add_argument("--scene_dir", type=str, default="/root/autodl-tmp/hai/VGGT-4D-baseline/data/monst3r",
                        help="Base directory containing dataset folders (e.g., ~/autodl-tmp/hai/VGGT-4D-baseline/data/monst3r)")
    
    args = parser.parse_args()
    
    base_path = Path(args.scene_dir).expanduser().resolve()
    process_all_datasets(base_path)


if __name__ == '__main__':
    main()