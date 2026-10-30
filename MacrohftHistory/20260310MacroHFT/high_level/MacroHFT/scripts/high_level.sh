nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' --exp reward1000 --epoch_number 20\
    >./logs/high_level/ETHUSDT/reward1000.log 2>&1 &


# nohup python -u RL/agent/high.py --dataset 'ETHUSDT' --device 'cuda:0' --exp test\
#     >./logs/high_level/ETHUSDT/test.log 2>&1 &