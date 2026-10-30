import torch.distributed.rpc as rpc
import time
import threading
import numpy as np
from collections import deque
from typing import Deque, Dict, List, Tuple
import random
from config import replay_buffer_sample_batch_size,replay_buffer_capacity,action_dim,n_state_1_dim,n_state_2_dim,seed
import argparse
import pathlib
import sys


ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".") 
from RL.util.logging_config import setup_logger  # 修改日志配置导入

from RL.util.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer_High(object):
    def __init__(self, batch_size, buffer_size, state_dim, state_dim_2, action_dim):
        self.logger = setup_logger("replay_buffer_high_server", "distributed_dqn.log", role="ReplayBuffer_High")
        self.batch_size = batch_size
        self.buffer_capacity = buffer_size
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed) 
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

    def buffer_size(self):
        return self.current_size
    
    def sample_batch(self):
        index = np.random.randint(0, self.current_size, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] =  self.buffer[key][index] 
            else:
                batch[key] =  self.buffer[key][index] 

        return batch, None, None
    
    def __len__(self) -> int:
        return self.current_size


class PrioritizedReplayBuffer_High(ReplayBuffer_High):
    """Prioritized Replay buffer for high-level policy.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(self, batch_size, buffer_size, state_dim, state_dim_2, action_dim, alpha: float = 0.3):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer_High, self).__init__(batch_size, buffer_size, state_dim, state_dim_2, action_dim)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store_transition(self, state, state_trend, state_clf, previous_action, demo_action, action, reward, 
                                next_state, next_state_trend, next_state_clf, next_previous_action, next_demo_action, terminal, q_memory):
        """Store experience and priority."""
        # Store transition in the buffer
        super().store_transition(state, state_trend, state_clf, previous_action, demo_action, action, reward, 
                                next_state, next_state_trend, next_state_clf, next_previous_action, next_demo_action, terminal, q_memory)
        
        # Store priority in the trees
        # We use the maximum priority for new transitions
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_capacity
        
    def sample_batch(self, beta: float = 0.6):
        """Sample a batch of experiences based on priority."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        # Create batch
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] =  self.buffer[key][indices]
            else:
                batch[key] =  self.buffer[key][indices]
        
        # Calculate weights for importance-sampling correction
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return batch, weights, indices
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
 
    @staticmethod
    def run(rank, host, world_size):
        # 初始化RPC，确保指向正确的机器和端口
        print("replay_buffer run start")
        rpc.init_rpc(
            "replay_buffer",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{host}:29500"  # 独立端口避免冲突
            )
        ) 
        # 注册RPC 
        print("replay_buffer已启动")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("收到中断信号，ReplayBuffer停止运行")

        # 保持RPC运行
        rpc.shutdown()

# 全局变量用于存储远程对象的RRef
remote_obj_rref = None
remote_obj_lock = threading.Lock()

def get_remote_replay_buffer():
    """
    获取远程ReplayBuffer对象的RRef
    """
    global remote_obj_rref
    with remote_obj_lock:
        if remote_obj_rref is None:
            
            # 在RPC框架初始化后立即创建远程对象
            buffer = PrioritizedReplayBuffer_High(
                
                replay_buffer_sample_batch_size,
                replay_buffer_capacity,
                n_state_1_dim,
                n_state_2_dim,
                action_dim,
                
            )
            remote_obj_rref = buffer
        return remote_obj_rref
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.0.109", help="host for distributed training")
    parser.add_argument("--world_size", type=int, default=5, help="world size for distributed training")
    args = parser.parse_args()
    
    world_size = args.world_size
    host = args.host    
    PrioritizedReplayBuffer_High.run(1, host, world_size) 