nohup python -u archetype/train_stage2_selector.py \
  --dataset ETHUSDT --horizon 72 \
  --vq_ckpt ./result/archetype/ETHUSDT/vq/vq_archetype_final.pt \
  --out_dir ./result/archetype/ETHUSDT/stage2 \
  --device cuda:0 --steps 200000 --lr 3e-4 \
  --alpha_kl 0.1 --ent_coef 0.01 --vf_coef 0.5 \
  --commission_fee 0.0005 --reward_scale 1000 --max_holding 0.2 --action_mode both \
  > ./logs/archetype/stage2/train_selector_$(date +%Y%m%d_%H%M%S).log 2>&1 &