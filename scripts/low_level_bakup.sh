nohup python -u RL/agent/low_level.py --alpha 2 --exp 'exp2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_0 >./logs/low_level/ETHUSDT/slope_0.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_1 >./logs/low_level/ETHUSDT/slope_1.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 4 --exp 'exp2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_2 >./logs/low_level/ETHUSDT/slope_2.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 0 --exp 'exp2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_3 >./logs/low_level/ETHUSDT/slope_3.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 2 --exp 'exp2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_4 >./logs/low_level/ETHUSDT/slope_4.log 2>&1 &

nohup python -u RL/agent/low_level.py --alpha 4 --exp 'exp2' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --label label_0 >./logs/low_level/ETHUSDT/vol_0.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 4 --exp 'exp2' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --label label_1 >./logs/low_level/ETHUSDT/vol_1.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp2' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --label label_2 >./logs/low_level/ETHUSDT/vol_2.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp2' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --label label_3 >./logs/low_level/ETHUSDT/vol_3.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp2' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --label label_4 >./logs/low_level/ETHUSDT/vol_4.log 2>&1 &

