import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pathlib
import sys
ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")



from RL.util.graph_utils import plot_money_curve



def test_plot_money_curve():
    # 加载数据
    test_data_path = os.path.join("/home/lance/quant/aiwork/MacroHFT/data/ETHUSDT/df_test.feather")
    df = pd.read_feather(test_data_path)
    
    save_path = os.path.join("/home/lance/quant/aiwork/MacroHFT/result/high_level/ETHUSDT")
    trade_records = np.load(os.path.join(save_path, "trade_records.npy"), allow_pickle=True)
    value_history = np.load(os.path.join(save_path, "value_history.npy"), allow_pickle=True)
    
    # 调用绘图函数
    plot_money_curve(trade_records, value_history, df, save_path)
    
    # 保存图表
    # plt.savefig(os.path.join("/home/lance/quant/aiwork/MacroHFT/test", "money_curve.png"))
    # print("图表已保存到 test/money_curve.png")

if __name__ == "__main__":
    test_plot_money_curve()