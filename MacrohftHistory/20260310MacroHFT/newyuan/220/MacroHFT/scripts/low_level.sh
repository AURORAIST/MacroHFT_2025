nohup python -u RL/agent/low_level.py --lr 1e-4 --epoch_number 15 --alpha 1 --exp 223 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_1 >./logs/low_level/ETHUSDT/slope_1_223.log 2>&1 &

nohup python -u RL/agent/low_level.py --lr 1e-4 --epoch_number 15 --alpha 1 --exp 223 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_2 >./logs/low_level/ETHUSDT/slope_2_223.log 2>&1 &

nohup python -u RL/agent/low_level.py --lr 1e-4 --epoch_number 15 --alpha 1 --exp 223 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' --label label_3 >./logs/low_level/ETHUSDT/slope_3_223.log 2>&1 &

nohup python -u RL/agent/low_level.py --lr 1e-4 --epoch_number 15 --alpha 1 --exp 223 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --label label_1 >./logs/low_level/ETHUSDT/vol_1_223.log 2>&1 &
 
nohup python -u RL/agent/low_level.py --lr 1e-4 --epoch_number 15 --alpha 1 --exp 223 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --label label_2 >./logs/low_level/ETHUSDT/vol_2_223.log 2>&1 &

nohup python -u RL/agent/low_level.py --lr 1e-4 --epoch_number 15 --alpha 1 --exp 223 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' --label label_3 >./logs/low_level/ETHUSDT/vol_3_223.log 2>&1 &

