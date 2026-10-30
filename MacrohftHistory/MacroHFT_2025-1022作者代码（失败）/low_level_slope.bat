@echo off
REM low_level.bat - Execute training tasks with CUDA GPU (并行启动)

REM Run with alpha=1
start /b "train_alpha1" python -u RL\agent\low_level.py  --alpha 1 --clf=slope --dataset=ETHUSDT --label=label_1 --device=cuda:0 > logs\low_level\ETHUSDT\exp4\slope\label_1\1\train_1.log 2>&1

REM Run with alpha=4
start /b "train_alpha4" python -u RL\agent\low_level.py  --alpha 4 --clf=slope --dataset=ETHUSDT --label=label_2 --device=cuda:0 > logs\low_level\ETHUSDT\exp4\slope\label_2\4\train_2.log 2>&1

REM Run with alpha=0 (no KL loss)
start /b "train_alpha0" python -u RL\agent\low_level.py  --alpha 0 --clf=slope --dataset=ETHUSDT --label=label_3 --device=cuda:0 > logs\low_level\ETHUSDT\exp4\slope\label_3\0\train_3.log 2>&1

echo All tasks started in parallel
pause