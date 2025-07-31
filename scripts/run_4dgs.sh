#!/bin/bash

# 切换到指定目录
cd /root/autodl-tmp/hai/VGGT-4D-baseline/third_party/4DGaussians

# 检查参数
if [ -z "$1" ]; then
    echo "Usage: $0 <exp_name>"
    exit 1
fi

EXP_NAME=$1

# 设置基础路径
DATA_ROOT="./data/${EXP_NAME}"
LOG_ROOT="./logs/${EXP_NAME}"
OUTPUT_ROOT="./output/${EXP_NAME}"

# 定义数据集目录
DATASETS=("nerfie" "nvidia")

# 定义 GPU 列表
GPUS=(1 2 3)
NUM_GPUS=${#GPUS[@]}

# 获取所有场景
SCENES=()
for DATASET in "${DATASETS[@]}"; do
    DATASET_PATH="${DATA_ROOT}/${DATASET}"
    if [ -d "$DATASET_PATH" ]; then
        for SCENE in "$DATASET_PATH"/*; do
            if [ -d "$SCENE" ]; then
                SCENES+=("$DATASET/$(basename "$SCENE")")
            fi
        done
    fi
done

TOTAL_SCENES=${#SCENES[@]}
if [ $TOTAL_SCENES -eq 0 ]; then
    echo "No scenes found in datasets: ${DATASETS[*]}"
    exit 1
fi

echo "Found $TOTAL_SCENES scenes. Starting processing with $NUM_GPUS GPUs (one scene per GPU)..."
echo "Total scenes: $TOTAL_SCENES"
for i in "${!SCENES[@]}"; do
    echo "  $((i+1)). ${SCENES[$i]}"
done

# 创建一个函数来处理单个场景
process_scene() {
    local SCENE_REL_PATH=$1
    local GPU_ID=$2
    local SCENE_INDEX=$3
    
    local DATASET_NAME=$(dirname "$SCENE_REL_PATH")
    local SCENE_NAME=$(basename "$SCENE_REL_PATH")
    
    local LOG_DIR="$LOG_ROOT/$SCENE_REL_PATH"
    local MODEL_OUTPUT_DIR="$OUTPUT_ROOT/$SCENE_REL_PATH"
    mkdir -p "$LOG_DIR"
    
    echo "[GPU $GPU_ID] Starting processing: $SCENE_REL_PATH"
    echo "[GPU $GPU_ID] Model output will be saved to: $MODEL_OUTPUT_DIR"
    echo "[GPU $GPU_ID] Log files will be saved to: $LOG_DIR"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    
    # 训练
    echo "[GPU $GPU_ID] Training $SCENE_REL_PATH ..."
    python train.py --port $((6017 + SCENE_INDEX)) \
        -s "$DATA_ROOT/$SCENE_REL_PATH" \
        --expname "$EXP_NAME/$SCENE_REL_PATH" \
        --configs "arguments/$DATASET_NAME/$SCENE_NAME.py" \
        > "$LOG_DIR/train.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[GPU $GPU_ID] ERROR: Training failed for $SCENE_REL_PATH"
        return 1
    fi
    echo "[GPU $GPU_ID] Training completed for $SCENE_REL_PATH"
    
    # 渲染
    echo "[GPU $GPU_ID] Rendering $SCENE_REL_PATH ..."
    python render.py --model_path "$MODEL_OUTPUT_DIR/" \
        --skip_train \
        --configs "arguments/$DATASET_NAME/$SCENE_NAME.py" \
        > "$LOG_DIR/render.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[GPU $GPU_ID] ERROR: Rendering failed for $SCENE_REL_PATH"
        return 1
    fi
    echo "[GPU $GPU_ID] Rendering completed for $SCENE_REL_PATH"
    
    # 评估
    echo "[GPU $GPU_ID] Evaluating $SCENE_REL_PATH ..."
    python metrics.py --model_path "$MODEL_OUTPUT_DIR/" \
        > "$LOG_DIR/metrics.log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[GPU $GPU_ID] ERROR: Metrics failed for $SCENE_REL_PATH"
        return 1
    fi
    echo "[GPU $GPU_ID] Evaluation completed for $SCENE_REL_PATH"
    
    echo "[GPU $GPU_ID] COMPLETED: $SCENE_REL_PATH"
    return 0
}

# 导出变量和函数
export -f process_scene
export DATA_ROOT LOG_ROOT OUTPUT_ROOT

# 控制并发：同时只运行3个场景（每个GPU一个）
RUNNING_PIDS=()
ASSIGNED_GPUS=()

echo "Starting parallel processing..."

# 初始化：启动前3个场景
for (( i=0; i<NUM_GPUS && i<TOTAL_SCENES; i++ )); do
    SCENE_REL_PATH="${SCENES[$i]}"
    GPU_ID=${GPUS[$i]}
    
    echo "Launching scene $((i+1))/$TOTAL_SCENES: $SCENE_REL_PATH on GPU $GPU_ID"
    process_scene "$SCENE_REL_PATH" "$GPU_ID" "$i" &
    RUNNING_PIDS+=($!)
    ASSIGNED_GPUS+=($i)
done

# 处理剩余场景
NEXT_SCENE_INDEX=$NUM_GPUS

# 监控并启动新任务
while [ ${#RUNNING_PIDS[@]} -gt 0 ]; do
    # 检查是否有任务完成
    for i in "${!RUNNING_PIDS[@]}"; do
        PID=${RUNNING_PIDS[$i]}
        if ! kill -0 $PID 2>/dev/null; then
            # 任务已完成
            wait $PID
            FINISHED_INDEX=${ASSIGNED_GPUS[$i]}
            echo "Scene on GPU ${GPUS[$FINISHED_INDEX]} finished"
            
            # 移除已完成的任务
            unset RUNNING_PIDS[$i]
            unset ASSIGNED_GPUS[$i]
            RUNNING_PIDS=("${RUNNING_PIDS[@]}")
            ASSIGNED_GPUS=("${ASSIGNED_GPUS[@]}")
            
            # 如果还有未处理的场景，启动新任务
            if [ $NEXT_SCENE_INDEX -lt $TOTAL_SCENES ]; then
                SCENE_REL_PATH="${SCENES[$NEXT_SCENE_INDEX]}"
                GPU_ID=${GPUS[$FINISHED_INDEX]}
                
                echo "Launching scene $((NEXT_SCENE_INDEX+1))/$TOTAL_SCENES: $SCENE_REL_PATH on GPU $GPU_ID"
                process_scene "$SCENE_REL_PATH" "$GPU_ID" "$NEXT_SCENE_INDEX" &
                RUNNING_PIDS+=($!)
                ASSIGNED_GPUS+=($FINISHED_INDEX)
                NEXT_SCENE_INDEX=$((NEXT_SCENE_INDEX + 1))
            fi
            break
        fi
    done
    
    # 短暂休眠避免过度占用CPU
    sleep 2
done

echo "All scenes processed successfully!"