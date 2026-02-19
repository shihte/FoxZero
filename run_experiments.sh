#!/bin/bash

# 1. 清除舊權重 (防止繼承中毒的模型)
rm -f foxzero_model.pth foxzero_weights_sync.pth
rm -f gpu*.log

echo "已清除舊權重。開始並行實驗..."

# 實驗 A (GPU 0): 穩健基準組
# 策略: 較高的 MCTS 模擬數 (200)，確保數據品質，Batch Size 較小以穩定更新。
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 foxzero/train_parallel.py \
  --actors 6 \
  --sims 200 \
  --batch_size 64 \
  --max_updates 5000 > gpu0_baseline.log 2>&1 &
PID0=$!
echo "實驗 A (GPU 0) 啟動 PID: $PID0 | Log: gpu0_baseline.log"

# 實驗 B (GPU 1): 極速產出組 (針對數據飢荒)
# 策略: 犧牲 MCTS 品質 (50 Sims)，極大化數據產出速度，使用大 Batch 吸收數據。
# 注意: 這裡 actors 開到 8 是為了壓榨 CPU
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 foxzero/train_parallel.py \
  --actors 8 \
  --sims 50 \
  --batch_size 128 \
  --max_updates 10000 > gpu1_fast.log 2>&1 &
PID1=$!
echo "實驗 B (GPU 1) 啟動 PID: $PID1 | Log: gpu1_fast.log"

# 實驗 C (GPU 2): 大吞吐量探索組
# 策略: 折衷的模擬數 (100)，但使用巨大的 Batch Size (256) 來穩定梯度方向。
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 foxzero/train_parallel.py \
  --actors 6 \
  --sims 100 \
  --batch_size 256 \
  --max_updates 5000 > gpu2_explore.log 2>&1 &
PID2=$!
echo "實驗 C (GPU 2) 啟動 PID: $PID2 | Log: gpu2_explore.log"

echo "---------------------------------------------------"
echo "監控指令 (即時查看 Buffer 成長):"
echo "tail -f gpu*.log | grep 'Buffer'"