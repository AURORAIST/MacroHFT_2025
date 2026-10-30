nohup python -u archetype/train_stage3_adapter.py \
  --dataset ETHUSDT --horizon 72 \
  --vq_ckpt ./result/archetype/ETHUSDT/vq/vq_archetype_final.pt \
  --selector_ckpt ./result/archetype/ETHUSDT/stage2/selector.pt \
  --out_dir ./result/archetype/ETHUSDT/stage3 \
  --device cuda:0 --steps 200000 --lr 3e-4 --gamma 0.99 \
  --beta1 1.0 --beta2 0.1 --ent_coef 0.01 \
  --commission_fee 0.0005 --reward_scale 1000 --max_holding 0.2 --action_mode both \
  > ./logs/archetype/stage3/train_adapter_$(date +%Y%m%d_%H%M%S).log 2>&1 &
