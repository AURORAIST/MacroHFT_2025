#失败
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 4 --exp 'short_action_3' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode short --action_size 1 --label label_1 --reward_no_action False >./logs/low_level/ETHUSDT/vol_1_short.log 2>&1 &
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'short_action_3' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode short --action_size 1 --label label_2 --reward_no_action False >./logs/low_level/ETHUSDT/vol_2_short.log 2>&1 &
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'short_action_3' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode short --action_size 1 --label label_3 --reward_no_action False >./logs/low_level/ETHUSDT/vol_3_short.log 2>&1 &

nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 1 --exp 'short_action_3' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode short --action_size 1 --label label_1 --reward_no_action False >./logs/low_level/ETHUSDT/slope_1_short.log 2>&1 &
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 4 --exp 'short_action_3' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode short --action_size 1 --label label_2 --reward_no_action False >./logs/low_level/ETHUSDT/slope_2_short.log 2>&1 &
nohup python -u RL/agent/low_level.py --lr 0.0001 --alpha 0 --exp 'short_action_3' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode short --action_size 1 --label label_3 --reward_no_action False >./logs/low_level/ETHUSDT/slope_3_short.log 2>&1 &

