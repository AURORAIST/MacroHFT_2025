0)（可选）清理旧结果
如果你想完全从头来（会删除旧 checkpoint/结果）：
rm -rf ./result/archetype/ETHUSDT/vq \
       ./result/archetype/ETHUSDT/stage2 \
       ./result/archetype/ETHUSDT/stage3
1) Stage 1：生成 DP demonstrations + 训练 VQ
1.1 生成 demo 数据集（DP）
python archetype/build_demo_dataset.py \
  --dataset ETHUSDT \
  --horizon 72 \
  --n_traj 30000 \
  --commission_fee 0.0005 \
  --gamma 0.999 \
  --reward_scale 1000 \
  --action_mode both \
  --max_holding 0.2 \
  --out ./data/ETHUSDT/archetype/demo_dp_h72_n30000.npz
1.2 训练 VQ（得到 vq_archetype_final.pt）
python archetype/train_vq.py \
  --dataset ETHUSDT \
  --npz ./data/ETHUSDT/archetype/demo_dp_h72_n30000.npz \
  --out_dir ./result/archetype/ETHUSDT/vq \
  --device cuda:0 \
  --epochs 20 \
  --batch_size 256 \
  --lr 3e-4 \
  --codebook_size 32 \
  --hidden_dim 128 \
  --z_dim 64 \
  --beta 0.25
2) Stage 2：训练 macro selector（带 demo consistency CE）
你刚加的诊断会在日志里输出：
python archetype/train_stage2_selector.py \
  --dataset ETHUSDT \
  --horizon 72 \
  --vq_ckpt ./result/archetype/ETHUSDT/vq/vq_archetype_final.pt \
  --out_dir ./result/archetype/ETHUSDT/stage2 \
  --device cuda:0 \
  --steps 200000 \
  --lr 3e-4 \
  --alpha_kl 0.1 \
  --ent_coef 0.01 \
  --vf_coef 0.5 \
  --commission_fee 0.0005 \
  --reward_scale 1000 \
  --max_holding 0.2 \
  --action_mode both
demoFail/failRate
demoTop5
actTop5
3) Stage 3：训练 micro adapter（含 Eq.(9) step-level CE）
python archetype/train_stage3_adapter.py \
  --dataset ETHUSDT \
  --horizon 72 \
  --vq_ckpt ./result/archetype/ETHUSDT/vq/vq_archetype_final.pt \
  --selector_ckpt ./result/archetype/ETHUSDT/stage2/selector.pt \
  --out_dir ./result/archetype/ETHUSDT/stage3 \
  --device cuda:0 \
  --steps 200000 \
  --lr 3e-4 \
  --gamma 0.99 \
  --beta1 1.0 \
  --beta2 0.1 \
  --ent_coef 0.01 \
  --commission_fee 0.0005 \
  --reward_scale 1000 \
  --max_holding 0.2 \
  --action_mode both