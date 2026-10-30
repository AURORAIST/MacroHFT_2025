import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import deque
import logging
import os
log = logging.getLogger(__name__)

def plot_money_curve(trade_records, value_history, df, save_path):
    """
    绘制图表
    - df价格曲线， 
    - 资金价值曲线 
    - 如果发生了买入，价格曲线的点击变红，卖出变蓝
    - timestamp 是在3个数据集中都存在
    参数:
      trade_records   买卖记录，不是每个时间都有买卖记录
      value_history   资金历史记录，
      df              价格数据
    """
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(12, 6))
 
    # 确保trade_records是DataFrame格式
    if isinstance(trade_records, list):
        trade_records = pd.DataFrame(trade_records)
    elif isinstance(trade_records, np.ndarray):
        trade_records = pd.DataFrame(trade_records.tolist())
         
    print(trade_records)
    # 绘制价格曲线
    ax1.plot(df['timestamp'], df['close'], label='价格', color='gray')
    ax1.set_xlabel('time')
    ax1.set_ylabel('price', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    
    # 在价格曲线上标记买卖点
    if not trade_records.empty:
        # 买入点标记为红色
        buy_records = trade_records[trade_records['type'] == 'buy']
        if not buy_records.empty:
            ax1.scatter(buy_records['datetime'], buy_records['price'], 
                       color='red', label='buy', s=50, alpha=0.7)
        
        # 卖出点标记为蓝色
        sell_records = trade_records[trade_records['type'] == 'sell']
        if not sell_records.empty:
            ax1.scatter(sell_records['datetime'], sell_records['price'], 
                       color='blue', label='sell', s=50, alpha=0.7)
    
    # 创建第二个y轴用于资金价值
    ax2 = ax1.twinx()
    
    # 处理value_history数据
    if isinstance(value_history, list):
        value_history = pd.DataFrame(value_history)
    elif isinstance(value_history, np.ndarray):
        value_history = pd.DataFrame(value_history.tolist())
    
    
    # 绘制资金价值曲线
    ax2.plot(value_history.iloc[:, 0], value_history.iloc[:, 1], 
                     label='value', color='green')
    
    ax2.set_ylabel('value', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 添加标题
    plt.title('price and value')
    
    # 调整布局
    fig.tight_layout()
    
    # 显示图表
    # plt.show()
    
    plt.savefig(os.path.join(save_path, "money_curve.png"))
    