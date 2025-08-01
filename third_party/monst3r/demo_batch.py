#!/usr/bin/env python3
# --------------------------------------------------------
# batch processing demo
# --------------------------------------------------------

import argparse
import os
import torch
import tempfile
import functools
import sys

# Make sure we import from the local demo.py, not from croco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import get_reconstructed_scene
from demo import get_args_parser as demo_get_args_parser
def find_scene_dirs(root_dir):
    """
    Recursively find all directories that contain an 'images' subdirectory
    """
    scene_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if 'images' in dirs:
            # Check if images directory is not empty
            images_path = os.path.join(root, 'images')
            if os.path.exists(images_path) and os.listdir(images_path):
                scene_dirs.append(root)
    return scene_dirs


def process_scene(args, model, device, image_size, scene_dir):
    """
    Process a single scene directory
    """
    # Import inside function to avoid conflicts
    
    # Determine output directory - create an 'output' directory at the same level as 'images'
    output_dir = os.path.join(scene_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set args.output_dir to the output directory
    args.output_dir = output_dir
    
    # Get sequence name (basename of scene_dir)
    seq_name = os.path.basename(scene_dir)
    
    # Get input files
    input_dir = os.path.join(scene_dir, 'images')
    input_files = [
        os.path.join(input_dir, fname)
        for fname in sorted(os.listdir(input_dir))
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not input_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Processing {len(input_files)} images from {input_dir}")
    
    # Create reconstruction function
    recon_fun = functools.partial(
        get_reconstructed_scene,
        args,
        output_dir,
        model,
        device,
        args.silent,
        image_size,
    )
    
    try:
        # Process the scene
        scene, outfile, imgs = recon_fun(
            filelist=input_files,
            schedule="linear",
            niter=300,
            min_conf_thr=1.1,
            as_pointcloud=True,
            mask_sky=False,
            clean_depth=True,
            transparent_cams=False,
            cam_size=0.05,
            show_cam=True,
            scenegraph_type="swinstride",
            winsize=5,
            refid=0,
            seq_name=seq_name,
            new_model_weights=args.weights,
            temporal_smoothing_weight=0.01,
            translation_weight="1.0",
            shared_focal=True,
            flow_loss_weight=0.01,
            flow_loss_start_iter=0.1,
            flow_loss_threshold=25,
            use_gt_mask=args.use_gt_davis_masks,
            fps=args.fps,
            num_frames=args.num_frames,
        )
        
        print(f"Processing completed. Output saved in {output_dir}")
        return outfile
    except Exception as e:
        print(f"Error processing scene {scene_dir}: {str(e)}")
        return None


def get_args_parser():
    # Import inside function to avoid conflicts
    parser = demo_get_args_parser()
    # Add scene_dir argument
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Root directory to search for scenes (directories containing 'images' subdirectories)",
    )
    return parser


def main():
    # Import required modules
    from dust3r.model import AsymmetricCroCo3DStereo
    
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)
    
    # Find all scene directories
    scene_dirs = find_scene_dirs(args.scene_dir)
    
    if not scene_dirs:
        print(f"No scene directories found in {args.scene_dir}")
        return
    
    print(f"Found {len(scene_dirs)} scene directories:")
    for scene_dir in scene_dirs:
        print(f"  - {scene_dir}")
    
    # Process each scene
    for i, scene_dir in enumerate(scene_dirs):
        print(f"\nProcessing scene {i+1}/{len(scene_dirs)}: {scene_dir}")
        process_scene(args, model, args.device, args.image_size, scene_dir)


if __name__ == "__main__":
    main()