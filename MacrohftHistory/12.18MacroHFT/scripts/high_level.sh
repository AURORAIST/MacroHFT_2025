nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' --epoch_number 1 --exp best_modelreset1\
    >./logs/high_level/pbest_modelreset1.log 2>&1 &

nohup python -u RL/agent/high.py --dataset 'ETHUSDT' --device 'cuda:0' --epoch_number 1 --exp best_modelreward1000epo35test\
    >./logs/high_level/best_modelreward1000epo35test0.log 2>&1 &

nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' --epoch_number 1 --n_actions 2 --exp singlevol3\
    >./logs/high_level/singlevol3.log 2>&1 &