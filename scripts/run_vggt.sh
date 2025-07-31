# Process Data using VGGT
export CUDA_VISIBLE_DEVICES=0
cd /root/autodl-tmp/hai/VGGT-4D-baseline
python third_party/vggt/demo_colmap_batch.py --scene_dir /root/autodl-tmp/hai/VGGT-4D-baseline/data/vggt

# Run 4DGS for VGGT
cd /root/autodl-tmp/hai/VGGT-4D-baseline/third_party/4DGaussians