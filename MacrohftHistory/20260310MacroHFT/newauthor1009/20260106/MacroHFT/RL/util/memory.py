import torch
import torch.nn.functional as F

import pdb
from scipy.spatial.distance import cosine
import numpy as np
import pathlib
import sys
ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")
from model.net import *

# episodicmemory.py
# 功能说明：实现基于隐藏状态的表记忆模块，支持核方法查询和在线重编码功能
# 主要特性：
# - 使用自定义核函数计算隐藏状态相似度
# - 支持top-k最近邻Q值加权查询
# - 提供模型更新后的隐藏状态批量重编码能力

# def custom_kernel(h, hi):
#     """相似性建模
#     自定义核函数：计算两个向量之间的平方距离倒数
    
#     Args:
#         h (np.array): 当前向量
#         hi (np.array): 比较向量
    
#     Returns:
#         float: 相似度得分
#     """
#     squared_distance = np.sum((h - hi) ** 2)
#     return 1 / (squared_distance + 1e-3)

def custom_kernel(query_hidden_state, hidden_states):
    """相似性建模
    自定义核函数：计算两个向量之间的平方距离倒数
    
    Args:
        h (np.array): 当前向量
        hi (np.array): 比较向量
    
    Returns:
        float: 相似度得分
    """
    # 使用向量化操作计算所有核函数值
    squared_distances = np.sum((hidden_states - query_hidden_state) ** 2, axis=1)
    return 1 / (squared_distances + 1e-3)

class episodicmemory():
    def __init__(self, capacity, k, state_dim, state_dim_2, hidden_dim, device):
        """
        初始化表记忆模块
        
        Args:
            capacity (int): 缓冲区容量
            k (int): 查询时使用的近邻数量
            state_dim (int): 单一状态维度
            state_dim_2 (int): 趋势状态维度
            hidden_dim (int): 隐藏状态维度
            device (torch.device): 计算设备（CPU/GPU）
        """
        self.capacity = capacity
        self.k = k
        self.current_size = 0
        self.count = 0
        self.device = device
        self.buffer = {"single_state": np.zeros((self.capacity, state_dim)),
                        "trend_state": np.zeros((self.capacity, state_dim_2)),
                        "previous_action": np.zeros((self.capacity)),
                        "hidden_state": np.zeros((self.capacity, hidden_dim)),
                        "action": np.zeros((self.capacity)),
                        "q_value": np.zeros((self.capacity))
                       }

    def add(self, hidden_state, action, q_value, single_state, trend_state, previous_action):
        """
        添加新经验到缓冲区
        
        Args:
            hidden_state (np.array): 隐藏状态
            action (int): 执行的动作
            q_value (float): 对应Q值
            single_state (np.array): 单一状态
            trend_state (np.array): 趋势状态
            previous_action (int): 前一个动作
        """
        self.buffer["single_state"][self.count] = single_state
        self.buffer["trend_state"][self.count] = trend_state
        self.buffer["previous_action"][self.count] = previous_action
        self.buffer["hidden_state"][self.count] = hidden_state
        self.buffer["action"][self.count] = action
        self.buffer["q_value"][self.count] = q_value
        self.count = (self.count + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def query(self, query_hidden_state, action):
        """
        查询与给定隐藏状态和动作匹配的加权Q值
        
        Args:
            query_hidden_state (np.array): 查询的隐藏状态
            action (int): 要查询的动作
        
        Returns:
            float: 加权Q值，若缓冲区未满返回NaN
        """
        if self.current_size != self.capacity:
            weighted_q_value = np.nan
        else:
            # 计算所有样本的核函数值
            #kernel_values = np.array([custom_kernel(query_hidden_state, hs) for hs in self.buffer["hidden_state"]])
            kernel_values = custom_kernel(query_hidden_state, self.buffer["hidden_state"])
            
            
            # 获取top-k索引
            top_k_indices = np.argsort(kernel_values)[-self.k:]
            # 获取top-k数据
            top_k_actions = self.buffer["action"][top_k_indices]
            top_k_q_values = self.buffer["q_value"][top_k_indices]
            
            # 创建掩码并计算权重
            mask = (top_k_actions == action).astype(float)
            weights = kernel_values[top_k_indices] / np.sum(kernel_values[top_k_indices])
            masked_weights = weights * mask
            normalized_weights = masked_weights / np.sum(masked_weights)
            # 返回加权Q值
            weighted_q_value = np.dot(normalized_weights, top_k_q_values)

        return weighted_q_value

    def re_encode(self, model):
        """
        使用指定模型重新编码缓冲区中的隐藏状态
        
        Args:
            model (nn.Module): 用于编码的神经网络模型
        """
        batch_size = 512
        for i in range(0, self.capacity, batch_size):
            batch_end = min(i + batch_size, self.capacity)
            # 转换数据为tensor
            single_states_batch = torch.tensor(self.buffer["single_state"][i:batch_end], dtype=torch.float32).to(self.device)
            trend_states_batch = torch.tensor(self.buffer["trend_state"][i:batch_end], dtype=torch.float32).to(self.device)
            previous_actions_batch = torch.tensor(self.buffer["previous_action"][i:batch_end], dtype=torch.long).to(self.device)
            # 使用模型进行编码
            with torch.no_grad():
                updated_hidden_states = model.encode(single_states_batch, trend_states_batch, previous_actions_batch).cpu().numpy()
            # 更新缓冲区
            self.buffer["hidden_state"][i:batch_end] = updated_hidden_states

