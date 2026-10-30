@echo off
REM low_level.bat - Execute training tasks with CUDA GPU

mkdir logs\low_level\ETHUSDT\exp4\vol\label_1\4
mkdir logs\low_level\ETHUSDT\exp4\vol\label_2\1
mkdir logs\low_level\ETHUSDT\exp4\vol\label_3\1

start /b "train_alpha4" python -u RL\agent\low_level.py  --alpha 4 --clf=vol --dataset=ETHUSDT --label=label_1 --device=cuda:0 > logs\low_level\ETHUSDT\exp4\vol\label_1\4\train_1.log 2>&1

start /b "train_alpha1" python -u RL\agent\low_level.py  --alpha 1 --clf=vol --dataset=ETHUSDT --label=label_2 --device=cuda:0 > logs\low_level\ETHUSDT\exp4\vol\label_2\1\train_2.log 2>&1

start /b "train_alpha1" python -u RL\agent\low_level.py  --alpha 1 --clf=vol --dataset=ETHUSDT --label=label_3 --device=cuda:0 > logs\low_level\ETHUSDT\exp4\vol\label_3\1\train_3.log 2>&1

echo All tasks completed
pause