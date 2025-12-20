nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' \
    >./logs/high_level/ETHUSDT.log 2>&1 &


    # python -u RL\agent\low_level.py --alpha 1 --clf=slope --dataset=ETHUSDT --label=label_1 --device=cuda:0