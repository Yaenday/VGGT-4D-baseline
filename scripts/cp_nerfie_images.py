import os
import shutil
from pathlib import Path

def copy_first_n_images(n=24):
    """
    Copy the first n images from each dataset in nerfie directory
    to a new nerfie_n directory, maintaining the same structure.
    
    Args:
        n (int): Number of images to copy from each dataset
    """
    # Define source and destination directories with correct path
    src_root = Path("data/raw_images/nerfie")
    dst_root = Path(f"data/raw_images/nerfie_{n}")
    
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
            
            # Take only the first n images
            first_n_images = image_files[:n]
            
            print(f"Copying {len(first_n_images)} images from {dataset_dir.name}...")
            
            # Copy each of the first n images
            for img_file in first_n_images:
                dst_file = dst_img_dir / img_file.name
                shutil.copy2(img_file, dst_file)
                print(f"Copied {img_file} to {dst_file}")

if __name__ == "__main__":
    copy_first_n_images(24)
    print("Finished copying first 24 images from each dataset.")