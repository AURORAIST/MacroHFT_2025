import torch
import random
import numpy as np
from collections import deque
from torch.utils.data import Dataset, DataLoader
import pdb


# 包含状态、动作、奖励等强化学习所需的基本元素
# 主要功能：存储转移信息(store_transition)，随机采样批次数据(sample)

class ReplayBuffer(object):
    def __init__(self, args, state_dim, state_dim_2, action_dim):
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_size
        self.seed = args.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.current_size = 0
        self.count = 0
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        self.buffer = {"state": np.zeros((self.buffer_capacity, state_dim)),
                        "state_trend": np.zeros((self.buffer_capacity, state_dim_2)),
                        "previous_action": np.zeros((self.buffer_capacity)),
                        "demo_action": np.zeros((self.buffer_capacity, action_dim)),
                        "action": np.zeros((self.buffer_capacity, 1)),
                        "reward": np.zeros(self.buffer_capacity),
                        "next_state": np.zeros((self.buffer_capacity, state_dim)),
                        "next_state_trend": np.zeros((self.buffer_capacity, state_dim_2)),
                        "next_previous_action": np.zeros((self.buffer_capacity)),
                        "next_demo_action": np.zeros((self.buffer_capacity, action_dim)),
                        "terminal": np.zeros(self.buffer_capacity),
                       }

    def store_transition(self, state, state_trend, previous_action, demo_action, action, reward, next_state, next_state_trend, next_previous_action, next_demo_action, terminal):
        self.buffer["state"][self.count] = state
        self.buffer["state_trend"][self.count] = state_trend
        self.buffer["previous_action"][self.count] = previous_action
        self.buffer["demo_action"][self.count] = demo_action
        self.buffer["action"][self.count] = action
        self.buffer["reward"][self.count] = reward
        self.buffer["next_state"][self.count] = next_state
        self.buffer["next_state_trend"][self.count] = next_state_trend
        self.buffer["next_previous_action"][self.count] = next_previous_action
        self.buffer["next_demo_action"][self.count] = next_demo_action
        self.buffer["terminal"][self.count] = terminal
        self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self):
        index = np.random.randint(0, self.current_size, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.float32)

        return batch, None, None


class ReplayBuffer_High(object):
    def __init__(self, args, state_dim, state_dim_2, action_dim):
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_size
        self.seed = args.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.current_size = 0
        self.count = 0
        self.buffer = {"state": np.zeros((self.buffer_capacity, state_dim)),
                        "state_trend": np.zeros((self.buffer_capacity, state_dim_2)),
                        "state_clf": np.zeros((self.buffer_capacity, 2)), # 分类状态
                        "previous_action": np.zeros((self.buffer_capacity)),
                        "demo_action": np.zeros((self.buffer_capacity, action_dim)),
                        "action": np.zeros((self.buffer_capacity, 1)),
                        "reward": np.zeros(self.buffer_capacity),
                        "next_state": np.zeros((self.buffer_capacity, state_dim)),
                        "next_state_trend": np.zeros((self.buffer_capacity, state_dim_2)),
                        "next_state_clf": np.zeros((self.buffer_capacity, 2)),  # 下一分类状态
                        "next_previous_action": np.zeros((self.buffer_capacity)),
                        "next_demo_action": np.zeros((self.buffer_capacity, action_dim)),
                        "terminal": np.zeros(self.buffer_capacity),
                        "q_memory": np.zeros(self.buffer_capacity),  # Q值记忆
                       }

    def store_transition(self, state, state_trend, state_clf, previous_action, demo_action, action, reward, 
                                next_state, next_state_trend, next_state_clf, next_previous_action, next_demo_action, terminal, q_memory):
        self.buffer["state"][self.count] = state
        self.buffer["state_trend"][self.count] = state_trend
        self.buffer["state_clf"][self.count] = state_clf
        self.buffer["previous_action"][self.count] = previous_action
        self.buffer["demo_action"][self.count] = demo_action
        self.buffer["action"][self.count] = action
        self.buffer["reward"][self.count] = reward
        self.buffer["next_state"][self.count] = next_state
        self.buffer["next_state_trend"][self.count] = next_state_trend
        self.buffer["next_state_clf"][self.count] = next_state_clf
        self.buffer["next_previous_action"][
            self.count] = next_previous_action
        self.buffer["next_demo_action"][self.count] = next_demo_action
        self.buffer["terminal"][self.count] = terminal
        self.buffer["q_memory"][self.count] = q_memory
        self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self):
        index = np.random.randint(0, self.current_size, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.float32)

        return batch, None, None