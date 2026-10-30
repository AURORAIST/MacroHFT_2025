nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' --exp exp2epoch30test\
    >./logs/high_level/ETHUSDTepo30test.log 2>&1 &


    # python -u RL\agent\low_level.py --alpha 1 --clf=slope --dataset=ETHUSDT --label=label_1 --device=cuda:0