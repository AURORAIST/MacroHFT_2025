
#nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp4_1' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_2 >./logs/low_level/ETHUSDT/slope_1.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp4_2' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.00002 --label label_2 >./logs/low_level/ETHUSDT/slope_4_2.log 2>&1 &
nohup python -u RL/agent/low_level.py --alpha 1 --exp 'exp4_3' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.00000 --label label_2 >./logs/low_level/ETHUSDT/slope_4_3.log 2>&1 &
#nohup python -u RL/agent/low_level.py --alpha 3 --exp 'exp4_3' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_2 >./logs/low_level/ETHUSDT/slope_3.log 2>&1 &
# nohup python -u RL/agent/low_level.py --alpha 4 --exp 'exp4_4' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_1 >./logs/low_level/ETHUSDT/slope_4.log 2>&1 &
# nohup python -u RL/agent/low_level.py --alpha 0 --exp 'exp4_0' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_1 >./logs/low_level/ETHUSDT/slope_0.log 2>&1 &
