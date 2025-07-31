#!/usr/bin/env python3
"""
Script to copy the first N images from each dataset in nerfie directory
to a new nerfie_n directory, maintaining the same structure.
Supports configurable parameters for number of images, random sampling, and threshold.
"""

import os
import shutil
import random
from pathlib import Path

# Global parameters - adjust these as needed
N = 24  # Number of images to copy from each dataset
RANDOM_PICK = True  # Whether to randomly pick images
END_THRESHOLD = 100  # When RANDOM_PICK is True, sample from the first END_THRESHOLD images

def copy_first_n_images(n=N, random_pick=RANDOM_PICK, end_threshold=END_THRESHOLD):
    """
    Copy n images from each dataset in nerfie directory
    to a new nerfie_n directory, maintaining the same structure.
    
    Args:
        n (int): Number of images to copy from each dataset
        random_pick (bool): Whether to randomly pick images
        end_threshold (int): When random_pick is True, sample from the first end_threshold images
    """
    # Define source and destination directories with correct path
    src_root = Path("data/raw_images/nerfie_ori")
    dst_root = Path(f"data/raw_images/nerfie")
    
    # Check if source directory exists
    if not src_root.exists():
        print(f"Error: Source directory {src_root} does not exist.")
        return
    
    # Create destination root if it doesn't exist
    dst_root.mkdir(parents=True, exist_ok=True)
    
    # Iterate through each dataset in the nerfie directory
    for dataset_dir in src_root.iterdir():
        if dataset_dir.is_dir():
            # Define source and destination image directories
            src_img_dir = dataset_dir / "images"
            dst_img_dir = dst_root / dataset_dir.name / "images"
            
            # Check if source image directory exists
            if not src_img_dir.exists():
                print(f"Warning: Image directory {src_img_dir} does not exist. Skipping...")
                continue
                
            # Create destination image directory
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all image files and sort them
            image_files = sorted(src_img_dir.glob("*.png"))
            
            # Select images based on parameters
            if random_pick:
                # Sample n images from the first end_threshold images
                if len(image_files) <= end_threshold:
                    selected_images = random.sample(image_files, min(n, len(image_files)))
                else:
                    candidate_images = image_files[:end_threshold]
                    selected_images = random.sample(candidate_images, min(n, len(candidate_images)))
                # Sort the randomly selected images to maintain order
                selected_images = sorted(selected_images)
            else:
                # Take only the first n images
                selected_images = image_files[:n]
            
            print(f"Copying {len(selected_images)} images from {dataset_dir.name}...")
            
            # Copy each selected image with sequential renaming
            for idx, img_file in enumerate(selected_images):
                # Create new filename with sequential numbering (000001, 000002, etc.)
                new_filename = f"{idx+1:06d}.png"
                dst_file = dst_img_dir / new_filename
                shutil.copy2(img_file, dst_file)
                print(f"Copied {img_file} to {dst_file}")

if __name__ == "__main__":
    copy_first_n_images()
    print(f"Finished copying images with parameters: N={N}, RANDOM_PICK={RANDOM_PICK}, END_THRESHOLD={END_THRESHOLD}")