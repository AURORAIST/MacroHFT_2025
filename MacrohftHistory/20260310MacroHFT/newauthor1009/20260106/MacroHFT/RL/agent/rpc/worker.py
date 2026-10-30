import torch
import torch.distributed.rpc as rpc
import numpy as np
from torch.distributed.rpc import RRef, rpc_async, remote
import time
import argparse
import pathlib
import sys
import random
ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".") 
from RL.agent.rpc.config import action_dim,seed,tech_indicator_list,tech_indicator_list_trend

from env.high_level_env import Training_Env
from RL.agent.rpc.replay_buffer_high_server import get_remote_replay_buffer
from RL.util.logging_config import setup_logger  # 修改日志配置导入
from RL.agent.rpc.parameter_server import get_remote_parameter_server
from model.net import *
from RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from RL.util.memory import episodicmemory

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Worker:
    def __init__(self, rank,    
                 parameter_server_ref:rpc.RRef, 
                 replay_buffer_ref:rpc.RRef, 
                 epsilon_start,
                 epsilon_end,
                 decay_length,
                 dataset, 
                 gamma, 
                 transcation_cost,
                 back_time_length,
                 local_buffer_size=32,     
                ):
        # 初始化日志记录器，添加角色信息
        self.logger = setup_logger(f"worker_{rank}", "distributed_dqn.log", role=f"Worker-{rank}")
        self.logger.info(f"Worker初始化，rank: {rank}, max_steps: {local_buffer_size}")
        
        self.rank = rank         
        self.gamma = gamma
        self.dataset=dataset
        if "BTC" in self.dataset:
            self.max_holding_number=0.01
        elif "ETH" in self.dataset:
            self.max_holding_number=0.2
        elif "DOT" in self.dataset:
            self.max_holding_number=10
        elif "LTC" in self.dataset:
            self.max_holding_number=10
        else:
            raise Exception ("we do not support other dataset yet")
 
        self.train_data_path = os.path.join(ROOT, "data", self.dataset, "whole")
        self.transcation_cost = transcation_cost
        self.back_time_length = back_time_length
        
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.clf_list = ['slope_360', 'vol_360']  # 趋势  波动 分类
        
        self.n_action = action_dim
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.slope_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).cpu()
        self.slope_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).cpu()
        self.slope_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).cpu()
        self.vol_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).cpu()
        self.vol_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).cpu()
        self.vol_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).cpu()        
        model_list_slope = [
            "./result/low_level/ETHUSDT/best_model/slope/1/best_model.pkl", 
            "./result/low_level/ETHUSDT/best_model/slope/2/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/slope/3/best_model.pkl"
        ]
        model_list_vol = [
            "./result/low_level/ETHUSDT/best_model/vol/1/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/vol/2/best_model.pkl",
            "./result/low_level/ETHUSDT/best_model/vol/3/best_model.pkl"
        ]
        self.slope_1.load_state_dict(torch.load(model_list_slope[0], map_location="cpu"))
        self.slope_2.load_state_dict(torch.load(model_list_slope[1], map_location="cpu"))
        self.slope_3.load_state_dict(torch.load(model_list_slope[2], map_location="cpu"))
        self.vol_1.load_state_dict(torch.load(model_list_vol[0], map_location="cpu"))
        self.vol_2.load_state_dict(torch.load(model_list_vol[1], map_location="cpu"))
        self.vol_3.load_state_dict(torch.load(model_list_vol[2], map_location="cpu"))
        self.slope_1.eval()
        self.slope_2.eval()
        self.slope_3.eval()
        self.vol_1.eval()
        self.vol_2.eval()
        self.vol_3.eval()
        
        self.slope_agents = {
            0: self.slope_1,
            1: self.slope_2,
            2: self.slope_3
        }
        self.vol_agents = {
            0: self.vol_1,
            1: self.vol_2,
            2: self.vol_3
        }
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32) 
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_length = decay_length
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end, decay_length=self.decay_length)
        self.epsilon = epsilon_start
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, "cpu")
        self.df = pd.read_feather(os.path.join(self.train_data_path, "train.feather") ).head(10000)
        self.local_buffer_size = local_buffer_size
          
        self.epoch_counter = 0
        self.step_count = 0
        self.episode_reward_sum = 0
        self.local_buffer = []
        self.parameter_server_ref:rpc.RRef = parameter_server_ref
        self.replay_buffer_ref:rpc.RRef = replay_buffer_ref
    
    def init_env(self):
        self.train_env = Training_Env(
                "full",
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=random.choices(range(self.n_action), k=1)[0],
                alpha = 0)
        return self.train_env    
    
    def select_action(self, state, state_trend, state_clf, info):
        """
        选择动作（训练时使用）
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            state_clf: 状态分类特征
            info: 包含历史动作等信息的字典
            
        Returns:
            int: 选择的动作编号 (0或1)
            
        Process:
            1. 以epsilon概率随机探索
            2. 否则通过超网络计算Q值并选择最优动作
        """
        # Convert input data to PyTorch tensors and move to target device
        # 转换输入数据为张量并移动到目标设备
        x1 = torch.FloatTensor(state).cpu()
        x2 = torch.FloatTensor(state_trend).cpu()
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).cpu()
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().cpu(), 0).cpu()
        
        # Epsilon-greedy action selection
        # epsilon-greedy策略选择动作
        if np.random.uniform() < (1-self.epsilon):
            # Get Q-values from slope/volatility agents
            # 获取斜率/波动率代理的Q值
            qs = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action)
            ]
            # Calculate hypernetwork output
            # 计算超网络输出 6个子代理的权重
            w = self.hyperagent(x1, x2, x3, previous_action)
            # Combine Q-values using hypernetwork weights
            # 使用超网络权重组合Q值
            actions_value = self.calculate_q(w, qs)
            # Select action with max Q-value
            # 选择最大Q值的动作 (根据动作q值，选择动作， max()[1]选择的数组下标,max()[0] q值)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action_choice = [0,1]
            action = random.choice(action_choice)
        return action
        
    def _push_local_buffer_to_global(self):
        """将本地经验池中的经验批量推送到全局经验池"""
        if not self.local_buffer:
            self.logger.debug("本地经验池为空，无需推送")
            return
        
        self.logger.info(f"推送{len(self.local_buffer)}条经验到全局经验池")
        # 批量推送经验到全局经验池
        for experience in self.local_buffer:
            single_state, trend_state, clf_state, previous_action, demo_action, action, reward, next_single_state, next_trend_state, next_clf_state, next_previous_action, next_demo_action, done, q_memory = experience
            self.replay_buffer_ref.rpc_sync().store_transition(single_state, trend_state, clf_state, previous_action, demo_action, action, reward, next_single_state, next_trend_state, next_clf_state, next_previous_action, next_demo_action, done, q_memory)
            
        
        # 清空本地经验池
        self.local_buffer.clear()
        self.logger.debug("本地经验池已清空")
        
        # 向ParameterServer报告本地批次完成
        self.parameter_server_ref.rpc_sync().report_local_batch_completion()
        self.logger.info("向ParameterServer报告本地批次完成")



    def calculate_hidden(self, state, state_trend, info):
        """
        提取隐藏状态表示
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            info: 包含历史动作等信息的字典
            
        Returns:
            numpy.ndarray: 隐藏状态的numpy数组表示
        """
        # Convert input data to PyTorch tensors and move to target device
        # 转换输入数据为张量并移动到目标设备
        x1 = torch.FloatTensor(state).cpu()
        x2 = torch.FloatTensor(state_trend).cpu()
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().cpu(), 0).cpu()
        with torch.no_grad():
            # Encode to get hidden state representation
            # 编码获取隐藏状态表示
            hs = self.hyperagent.encode(x1, x2, previous_action).cpu().numpy()
        return hs
    
    
    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)# qs将6个代理的2个动作的权重 [6，2]  => [6,1,2]
        q_tensor = q_tensor.permute(1, 0, 2) # 重排维度为 [1,6,2]
        weights_reshaped = w.view(-1, 1, 6) #  超代理评估的6个子代理权重为[1,6], 改变形状为[1，1，6]
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1) # 执行批量矩阵乘法[1,1,2]并压缩维度得到最终Q值 [1, 2]
        
        return combined_q
    def q_estimate(self, state, state_trend, state_clf, info):
        """
        状态价值估计
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            state_clf: 状态分类特征
            info: 包含历史动作等信息的字典
            
        Returns:
            float: 当前状态的最大Q值估计
        """
        # Convert input data to PyTorch tensors and move to target device
        # 转换输入数据为张量并移动到目标设备
        x1 = torch.FloatTensor(state).cpu()
        x2 = torch.FloatTensor(state_trend).cpu()
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).cpu()
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().cpu(),0).cpu()
        
        # Get Q-values from slope/volatility agents
        # 获取斜率/波动率代理的Q值
        
        qs = [
                self.slope_agents[0](x1, x2, previous_action),
                self.slope_agents[1](x1, x2, previous_action),
                self.slope_agents[2](x1, x2, previous_action),
                self.vol_agents[0](x1, x2, previous_action),
                self.vol_agents[1](x1, x2, previous_action),
                self.vol_agents[2](x1, x2, previous_action)
        ]
        # Calculate hypernetwork output
        # 计算超网络输出
        w = self.hyperagent(x1, x2, x3, previous_action)
        # 形状[1,6]， 6个 子网络的权重，
        # Combine Q-values using hypernetwork weights
        # 使用超网络权重组合Q值  
        actions_value = self.calculate_q(w, qs)
        # Return max Q-value as state estimate
        # 返回最大Q值作为状态估计
        q = torch.max(actions_value, 1)[0].detach().cpu().numpy()
        
        return q

    def store_experience(self,single_state, trend_state, clf_state, previous_action, demo_action, action, reward, next_single_state, next_trend_state, next_clf_state, next_previous_action, next_demo_action, done, q_memory):
 
        self.local_buffer.append(
                        (single_state, trend_state, clf_state, previous_action, demo_action, action, reward, 
                        next_single_state, next_trend_state, next_clf_state, next_previous_action, next_demo_action, 
                        done, q_memory)
                    )

    def explore(self, single_state, trend_state, clf_state, info):
        """主循环"""
        self.logger.info(f"Worker-{self.rank}开始探索")
        try:
            
            params = self.parameter_server_ref.rpc_sync().get_parameters()  
            
            for p, new_p in zip(self.hyperagent.parameters(), params):
                p.data.copy_(new_p)
                
            if self.step_count > 4320:
                # 定期重新编码记忆, 因为超代理的隐藏层训练后发生了变化，
                # Periodically re-encode memory
                self.logger.info(f"Worker-{self.rank}开始重新编码记忆")
                self.memory.re_encode(self.hyperagent)    
                self.logger.info(f"Worker-{self.rank}重新编码记忆完成")
                
            self.logger.info(f"Worker-{self.rank}参数更新完成")
            
            for step in range(self.local_buffer_size):

                # 选择动作
                if single_state is None:
                    self.logger.error("环境返回的初始状态为None")
                    raise ValueError("环境返回的初始状态为None")
                # 使用ε-greedy策略选择动作
                # Select action using ε-greedy strategy
                action = self.select_action(single_state, trend_state, clf_state, info)
                # 执行环境步进操作
                # Execute environment step
                next_single_state, next_trend_state, next_clf_state, reward, done, next_info = self.train_env.step(action)
                # 计算隐藏状态
                # Calculate hidden state
                hs = self.calculate_hidden(single_state, trend_state, info)
                # 计算目标Q值
                # Calculate target Q-value
                q = reward + self.gamma * (1 - done) * self.q_estimate(next_single_state, next_trend_state, next_clf_state, next_info)
                # 查询记忆库中的Q值
                # Query Q-value from memory
                q_memory = self.memory.query(hs, action)
                if np.isnan(q_memory):
                    q_memory = q
                # 获取历史动作信息
                # Get historical action information
                previous_action = info['previous_action']
                demo_action = info['q_value']
                next_previous_action = next_info['previous_action']
                next_demo_action = next_info['q_value']
                # 存储经验到回放缓冲区
                # Store transition in replay buffer
                self.store_experience(single_state, trend_state, clf_state, previous_action, demo_action, action, reward, next_single_state, next_trend_state, next_clf_state, next_previous_action, next_demo_action, done, q_memory)
                # 更新记忆库
                # Update memory
                self.memory.add(hs, action, q, single_state, trend_state, previous_action)
                self.episode_reward_sum += reward

                single_state, trend_state, clf_state, info = next_single_state, next_trend_state, next_clf_state, next_info
                
                self.step_count += 1
                
                
                if done:
                    self.logger.info(f"Episode完成，步数: {step+1}")
                    self.epsilon = self.epsilon_scheduler.get_epsilon(self.epoch_counter)
                    self.init_env()
                    self.logger.info("环境重新初始化")
                    single_state, trend_state, clf_state, info = self.train_env.reset()
                    self.logger.info("环境reset")
                    self.epoch_counter += 1
                    self.episode_reward_sum = 0
                    # 检查reset后的state是否为None
                    if single_state is None:
                        self.logger.error("环境reset后返回的状态为None")
                        raise ValueError("环境reset后返回的状态为None")

            # 运行结束时推送剩余的经验
            if self.local_buffer:
                self.logger.info("推送剩余经验到全局经验池")
                self._push_local_buffer_to_global()
                
            return single_state, trend_state, clf_state, info    
        except KeyboardInterrupt:
            self.logger.info(f"收到中断信号，Worker-{self.rank}停止运行")
        except Exception as e:
            self.logger.error(f"Worker-{self.rank}运行出错: {e}", exc_info=True)



    @staticmethod
    def run(rank, host, world_size ,
                epsilon_start,
                epsilon_end,
                decay_length,
                dataset, 
                gamma, 
                transcation_cost,
                back_time_length,
                local_buffer_size  ):
        print("worker run start")
        seed_torch(seed)
        rpc.init_rpc(
            f"worker_{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{host}:29500"  # Parameter Server运行在machine1上
            )
        )
        print("worker init")
        ps_info = rpc.get_worker_info("parameter_server")
        rb_info = rpc.get_worker_info("replay_buffer")
        parameter_server_ref:rpc.RRef = remote(ps_info, get_remote_parameter_server)
        replay_buffer_ref:rpc.RRef = remote(rb_info, get_remote_replay_buffer)
        worker = Worker(rank, parameter_server_ref, replay_buffer_ref,
                        epsilon_start,
                        epsilon_end,
                        decay_length,
                        dataset, 
                        gamma, 
                        transcation_cost,
                        back_time_length,
                        local_buffer_size
                        )
         
        train_env = worker.init_env()
        worker.logger.info("环境初始化")
        single_state, trend_state, clf_state, info = train_env.reset()
        worker.epoch_counter += 1
        print(parameter_server_ref)
        print(replay_buffer_ref)
        try:
            while True:
                # 等待学习完成
                worker.logger.info("等待学习完成")
                learner_status_flag = parameter_server_ref.rpc_sync().get_learning_status()
                if learner_status_flag:
                    time.sleep(2)
                else:
                    single_state, trend_state, clf_state, info = worker.explore(single_state, trend_state, clf_state, info)
            pass
        except KeyboardInterrupt:
            worker.logger.info("收到中断信号，Worker停止运行")
        finally:
            rpc.shutdown()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.0.109", help="host for distributed training")
    parser.add_argument("--node_rank", type=int, default=1, help="Node rank for distributed training")
    parser.add_argument("--world_size", type=int, default=5, help="world size for distributed training") 
    parser.add_argument("--gamma", type=float, default=0.99)  # 折扣因子 / Discount factor
    parser.add_argument("--epsilon_start",type=float,default=0.7)  # 初始探索率 / Initial exploration rate
    parser.add_argument("--epsilon_end",type=float,default=0.3)  # 最小探索率 / Minimum exploration rate
    parser.add_argument("--decay_length",type=int,default=5)  # 探索衰减周期 / Exploration decay length
    parser.add_argument("--dataset", type=str, default="ETHUSDT")  # 数据集名称 / Dataset name
    parser.add_argument("--transcation_cost", type=float, default=2.0 / 100000)  # 交易成本 / Transaction cost
    parser.add_argument("--back_time_length", type=int, default=1)  # 回滚时间长度 / Rollback time length
    parser.add_argument("--local_buffer_size", type=int, default=512)  # 本地经验缓冲区大小 / Local experience buffer size
    
    
    args = parser.parse_args()
    
    world_size = args.world_size
    host = args.host   
    worker_rank = args.node_rank + 3  
    Worker.run(worker_rank, host, world_size,
                args.epsilon_start,
                args.epsilon_end,
                args.decay_length,
                args.dataset, 
                args.gamma, 
                args.transcation_cost,
                args.back_time_length,
                args.local_buffer_size
               )