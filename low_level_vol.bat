@echo off
REM low_level.bat - Execute training tasks with CUDA GPU


REM Run with alpha=0 (no KL loss)
python -u RL\agent\low_level.py --alpha 4 --clf=vol --dataset=ETHUSDT --label=label_1 --device=cuda:0

REM Run with alpha=0 (no KL loss)
python -u RL\agent\low_level.py --alpha 1 --clf=vol --dataset=ETHUSDT --label=label_2 --device=cuda:0


REM Run with alpha=0 (no KL loss)
python -u RL\agent\low_level.py --alpha 1 --clf=vol --dataset=ETHUSDT --label=label_3 --device=cuda:0


echo All tasks completed
pause