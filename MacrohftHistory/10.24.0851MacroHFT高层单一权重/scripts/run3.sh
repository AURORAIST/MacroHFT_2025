nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' --k 4 --epoch_number 1\
    >./logs/high_level/ETHUSDT.log 2>&1 &
