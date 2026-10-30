mkdir -p ./logs/archetype/stage1 ./logs/archetype/stage2 ./logs/archetype/stage3

nohup python -u archetype/build_demo_dataset.py \
  --dataset ETHUSDT --horizon 72 --n_traj 30000 \
  --commission_fee 0.0005 --gamma 0.999 --reward_scale 1000 \
  --action_mode both --max_holding 0.2 \
  --out ./data/ETHUSDT/archetype/demo_dp_h72_n30000.npz \
  > ./logs/archetype/stage1/build_demo_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python -u archetype/train_vq.py \
  --dataset ETHUSDT \
  --npz ./data/ETHUSDT/archetype/demo_dp_h72_n30000.npz \
  --out_dir ./result/archetype/ETHUSDT/vq \
  --device cuda:0 --epochs 20 --batch_size 256 --lr 3e-4 \
  --codebook_size 32 --hidden_dim 128 --z_dim 64 --beta 0.25 \
  > ./logs/archetype/stage1/train_vq_$(date +%Y%m%d_%H%M%S).log 2>&1 &
