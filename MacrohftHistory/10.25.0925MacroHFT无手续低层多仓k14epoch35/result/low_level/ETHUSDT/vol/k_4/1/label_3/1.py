import numpy as np
import os

path = "MacroHFT/result/low_level/ETHUSDT/vol/k_4/1/label_3/action.npy"
actions = np.load(path, allow_pickle=True)
print("actions.shape:", actions.shape)

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "actions_all.txt")

with open(output_path, "w") as f:
    for episode in actions:
        f.write(" ".join(map(str, episode)) + "\n")
print(f"已写入 {output_path}")