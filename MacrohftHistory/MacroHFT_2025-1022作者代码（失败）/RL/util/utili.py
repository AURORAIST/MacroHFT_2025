import torch
import numpy as np
from collections import deque, namedtuple
import random
import pdb


def get_ada(ada,decay_freq=2,ada_counter=0, decay_coffient=0.5):
    """
    自适应参数衰减函数（指数衰减）
    
    参数:
        ada (float): 当前自适应参数值
        decay_freq (int): 衰减频率（每N步执行一次）
        ada_counter (int): 步数计数器
        decay_coffient (float): 衰减系数（0-1范围）
        
    返回:
        float: 衰减后的自适应参数值
    """
    if ada_counter % decay_freq==1:
        ada = decay_coffient*ada
    return ada


def get_epsilon( epsilon,max_epsilon=1, epsilon_counter=0, decay_freq=2,decay_coffient=0.5):
    """
    探索率增长函数（线性增长）
    
    参数:
        epsilon (float): 当前探索率值
        max_epsilon (float): 最大探索率阈值
        epsilon_counter (int): 探索步数计数器
        decay_freq (int): 增长频率（每N步执行一次）
        decay_coffient (float): 增长系数（0-1范围）
        
    返回:
        float: 更新后的探索率值
    """
    if epsilon_counter%decay_freq == 1:
        epsilon =epsilon+(max_epsilon-epsilon)*decay_coffient
    return epsilon

class LinearDecaySchedule(object):
    """
    线性衰减调度器
    
    功能: 管理探索率(epsilon)的线性衰减过程
    特性: 
        - 起始探索率到终止探索率的线性过渡
        - 支持设置衰减总步数
        
    属性:
        start_epsilon (float): 初始探索率
        end_epsilon (float): 最小探索率
        decay_length (int): 完整衰减周期对应的步数
    """
    def __init__(self, start_epsilon, end_epsilon, decay_length):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_length = decay_length

    def get_epsilon(self, t):
        """
        计算指定步数的探索率值
        
        参数:
            t (int): 当前步数
            
        返回:
            float: 衰减后的探索率值（不会低于end_epsilon）
        """
        return max(self.end_epsilon, self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (t / self.decay_length))
