nohup python -u RL/agent/low_level.py --alpha 1 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --k 1 --epoch_number 35\
    --label label_3 >./logs/low_level/ETHUSDT/volk1_3.log 2>&1 &

nohup python -u RL/agent/low_level.py --alpha 1 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --k 2 --epoch_number 35\
    --label label_3 >./logs/low_level/ETHUSDT/volk2_3.log 2>&1 &