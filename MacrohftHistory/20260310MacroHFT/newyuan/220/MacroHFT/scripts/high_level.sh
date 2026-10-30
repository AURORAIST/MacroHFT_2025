nohup python -u RL/agent/high_level.py \
    --epoch_number 5 \
    --dataset 'ETHUSDT' \
    --lr 1e-4 \
    --exp 223wueplison \
    --device 'cuda:0' \
    >./logs/high_level/223wueplison.log 2>&1 &
