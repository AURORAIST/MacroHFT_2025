nohup python -u RL/agent/low_level_eval_main.py  --exp 'exp_default_1' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode long --action_size 1 --label label_2 >./logs/low_level/ETHUSDT/slope_2_long.log 2>&1 &
nohup python -u RL/agent/low_level_eval_main.py  --exp 'exp_default_1' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode long --action_size 1 --label label_1 >./logs/low_level/ETHUSDT/slope_1_long.log 2>&1 &
nohup python -u RL/agent/low_level_eval_main.py  --exp 'exp_default_1' --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode long --action_size 1 --label label_3 >./logs/low_level/ETHUSDT/slope_3_long.log 2>&1 &

nohup python -u RL/agent/low_level_eval_main.py  --exp 'exp_default_1' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode long --action_size 1 --label label_2 >./logs/low_level/ETHUSDT/vol_2_long.log 2>&1 &
nohup python -u RL/agent/low_level_eval_main.py  --exp 'exp_default_1' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode long --action_size 1 --label label_1 >./logs/low_level/ETHUSDT/vol_1_long.log 2>&1 &
nohup python -u RL/agent/low_level_eval_main.py  --exp 'exp_default_1' --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --transcation_cost 0.0002 --action_mode long --action_size 1 --label label_3 >./logs/low_level/ETHUSDT/vol_3_long.log 2>&1 &

