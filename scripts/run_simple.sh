### plain 4DGaussians
conda activate 4DGS
cd /root/autodl-tmp/hai/VGGT-4D-baseline/third_party/4DGaussians
# training
python train.py -s  data/hypernerf/virg/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py
# rendering
python render.py --model_path "output/hypernerf/broom2/"  --skip_train --configs arguments/hypernerf/broom2.py
# evaluation
python metrics.py --model_path "output/hypernerf/broom2/"

### monst3r
conda activate 4DGS
cd /root/autodl-tmp/hai/VGGT-4D-baseline/third_party/monst3r
# inference
python demo.py --input demo_data/lady-running --output_dir demo_tmp --seq_name lady-running --not_batchify

export CUDA_VISIBLE_DEVICES=5
python demo.py --input /root/autodl-tmp/hai/VGGT-4D-baseline/data/monst3r/nvidia/Balloon1/images --output_dir /root/autodl-tmp/hai/VGGT-4D-baseline/data/monst3r/nvidia/Balloon1/output --seq_name Balloon1 --not_batchify


### VGGT
conda activate 4DGS
cd /root/autodl-tmp/hai/VGGT-4D-baseline/third_party/vggt

cd /root/autodl-tmp/hai/VGGT-4D-baseline/third_party/4DGaussians

python train.py --port 6017 \
    -s data/vggt/nvidia/Balloon1  \
    --expname "vggt/nvidia/Balloon1" \
    --configs arguments/nvidia/Balloon1.py

# rendering
python render.py --model_path "output/vggt/nvidia/Balloon1/"  --skip_train --configs arguments/nvidia/Balloon1.py
# evaluation
python metrics.py --model_path "output/vggt/nvidia/Balloon1/"


python train.py --port 6017 \
    -s /root/autodl-tmp/hai/VGGT-4D-baseline/data/vggt/nvidia/Umbrella  \
    --expname "vggt/nvidia/Umbrella" \
    --configs arguments/nvidia/Umbrella.py

# rendering
python render.py --model_path "output/vggt/nvidia/Umbrella/"  --skip_train --configs arguments/nvidia/Balloon1.py
# evaluation
python metrics.py --model_path "output/vggt/nvidia/Umbrella/"