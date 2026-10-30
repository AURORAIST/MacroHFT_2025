@echo off
REM low_level.bat - Execute training tasks with CUDA GPU (并行启动)

REM Run with alpha=4
start /b "train_alpha4" python -u RL\agent\low_level.py --epoch_number 20 --alpha 4 --clf=slope --dataset=ETHUSDT --label=label_2 --device=cpu > logs\low_level\ETHUSDT\exp4\slope\label_2\4\train_2.log 2>&1

start /b "train_alpha4" python -u RL\agent\low_level.py --epoch_number 20 --alpha 4 --clf=vol --dataset=ETHUSDT --label=label_1 --device=cuda:0 > logs\low_level\ETHUSDT\exp4\vol\label_1\4\train_1.log 2>&1

echo All tasks completed
pause
