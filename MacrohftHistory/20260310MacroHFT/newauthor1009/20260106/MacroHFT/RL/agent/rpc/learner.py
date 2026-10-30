import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote
import time
import random
import numpy as np
from torch.nn.utils import clip_grad_norm_
from typing import Dict, List, Tuple
import argparse
import pathlib
import sys

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".") 
from RL.agent.rpc.config import replay_buffer_sample_batch_size,action_dim,n_state_1_dim,n_state_2_dim,seed,tech_indicator_list,tech_indicator_list_trend


from RL.agent.rpc.replay_buffer_high_server import get_remote_replay_buffer
from RL.util.logging_config import setup_logger  # 修改日志配置导入
from RL.agent.rpc.parameter_server import get_remote_parameter_server
from model.net import *


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Learner:
    def __init__(self, 
                parameter_server_ref:rpc.RRef, 
                replay_buffer_ref:rpc.RRef,
                prior_eps,
                gamma,
                alpha,
                beta,
                tau,
                lr
                ):
        # 初始化日志记录器，添加角色信息
        self.logger = setup_logger("learner", "distributed_dqn.log", role="Learner")
        self.logger.info(f"Learner初始化，state_dim: {n_state_1_dim},{n_state_2_dim} action_dim: {action_dim}") 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")         
         
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.clf_list = ['slope_360', 'vol_360']  # 趋势  波动 分类
    
        self.n_action = action_dim
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.slope_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)        
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
        self.slope_1.load_state_dict(torch.load(model_list_slope[0], map_location=self.device))
        self.slope_2.load_state_dict(torch.load(model_list_slope[1], map_location=self.device))
        self.slope_3.load_state_dict(torch.load(model_list_slope[2], map_location=self.device))
        self.vol_1.load_state_dict(torch.load(model_list_vol[0], map_location=self.device))
        self.vol_2.load_state_dict(torch.load(model_list_vol[1], map_location=self.device))
        self.vol_3.load_state_dict(torch.load(model_list_vol[2], map_location=self.device))
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
        
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
 
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()        
        self.tau = tau
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.prior_eps = prior_eps
        self.batch_size = replay_buffer_sample_batch_size

        self.sample_beta = 0.6     
        self.learning_rounds = 20  # 每次学习10轮
        self.logger.info(f"Learner配置完成，device: {self.device}, batch_size: {self.batch_size}")
        self.parameter_server_ref:rpc.RRef = parameter_server_ref
        self.replay_buffer_ref:rpc.RRef = replay_buffer_ref
        
        self.train_step_count =0
        
    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)# qs将6个代理的2个动作的权重 [6，2]  => [6,1,2]
        q_tensor = q_tensor.permute(1, 0, 2) # 重排维度为 [1,6,2]
        weights_reshaped = w.view(-1, 1, 6) #  超代理评估的6个子代理权重为[1,6], 改变形状为[1，1，6]
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1) # 执行批量矩阵乘法[1,1,2]并压缩维度得到最终Q值 [1, 2]
        
        return combined_q      
     
    def train_step(self):
        self.logger.info("开始训练步骤")
        # 通知Worker暂停探索
        self.logger.debug("通知Worker暂停探索")
        self.parameter_server_ref.rpc_sync().set_learning_status(True)   
        
        # 进行10轮学习
        self.logger.info(f"开始{self.learning_rounds}轮学习")
        self.sample_beta = self.sample_beta+(1-self.sample_beta)*0.00001
        
        for i in range(self.learning_rounds):
            self.logger.debug(f"第{i+1}轮学习")
            
            loss = self.update_model() 
            # 记录损失日志
            self.logger.info(f"第{i+1}轮学习完成，损失: {loss}") 
            
        # 推送参数到ParameterServer
        self.logger.info("推送参数到ParameterServer")
        params = [p.detach().clone().cpu() for p in self.hyperagent.parameters()]
        self.parameter_server_ref.rpc_sync().update_parameters(params)    
        
        # 重置ParameterServer的批次完成计数
        self.logger.debug("重置ParameterServer的批次完成计数")
        self.parameter_server_ref.rpc_sync().reset_batch_completion()    
        # 清除学习状态
        self.logger.info("清除学习状态")
        self.parameter_server_ref.rpc_sync().set_learning_status(False)    
        self.train_step_count += 1
        self.logger.info("训练步骤完成")
        

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        # 从独立的ReplayBuffer采样
        samples, weights, indices= self.replay_buffer_ref.rpc_sync().sample_batch(self.sample_beta)      
        batch = {}
        for key in samples.keys():
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] = torch.tensor(samples[key], dtype=torch.long).to(self.device)
            else:
                batch[key] = torch.tensor(samples[key], dtype=torch.float32).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device) 
        
        # 1-step Learning loss
        loss , loss_for_prior = self._compute_dqn_loss(batch)
        
        # PER: importance sampling before average
        # loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.hyperagent.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities         
        new_priorities = loss_for_prior + self.prior_eps
        self.replay_buffer_ref.rpc_sync().update_priorities(indices, new_priorities)
        
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.logger.info("目标网络参数已更新")
        return loss.item()

    def _compute_dqn_loss(self,batch):
        
        
        """
        更新策略网络参数，包含TD误差、记忆误差和KL散度的联合优化
        
        Args:
            replay_buffer: 经验回放缓冲区，包含(s, a, r, s', demo_action)等数据
            
        Returns:
            tuple: 包含各损失项和Q值统计的元组
                - td_error: TD误差损失
                - memory_error: 记忆误差损失
                - KL_loss: KL散度损失
                - q_current_mean: 当前Q值均值
                - q_target_mean: 目标Q值均值
        """
        # Sample transition from replay buffer & move to device
        # 从经验回放缓冲区采样并移动到指定设备
        
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # Calculate current and target hypernetwork outputs
        # 计算当前和目标超网络输出
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
        w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])

        # Compute Q-values from slope/volatility agents
        # 计算斜率/波动率代理的Q值
        qs_current = [
                    self.slope_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.slope_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.slope_agents[2](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[2](batch['state'], batch['state_trend'], batch['previous_action'])
        ]
        qs_next = [
                    self.slope_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.slope_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.slope_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action'])
        ]
        # Calculate Q distribution and gather selected actions
        # 计算Q分布并收集选定动作
        q_distribution = self.calculate_q(w_current, qs_current)
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        # Compute target Q-values with double DQN style argmax
        # 使用Double DQN风格计算目标Q值
        a_argmax = self.calculate_q(w_next_, qs_next).argmax(dim=-1, keepdim=True)
        q_nexts = self.calculate_q(w_next, qs_next)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * q_nexts.gather(-1, a_argmax).squeeze(-1)
        # Calculate individual loss components
        # 计算各损失项
        td_error_for_prior = torch.abs(q_target - q_current).detach().cpu().numpy()
        td_error = self.loss_func(q_current, q_target)
        memory_error = self.loss_func(q_current, batch['q_memory'])
        # Demonstration-guided KL divergence loss
        # 示范引导的KL散度损失
        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )
        # Total loss with weighted components
        # 加权总损失，   alpha 代表了记忆的经验权重， beta代表先验q-table权重
        loss = td_error + self.alpha * memory_error + self.beta * KL_loss
        return loss , td_error_for_prior
             

    @staticmethod
    def run(rank, host, world_size ,
                prior_eps,
                gamma,
                alpha,
                beta,
                tau,
                lr):
        """主循环"""
        # 初始化RPC
        seed_torch(seed)
        print("learner run start")
        rpc.init_rpc(
            "learner",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{host}:29500"
            )
        )
        time.sleep(10) 
        print("learner init")
        # 创建Learner实例
        ps_info = rpc.get_worker_info("parameter_server")
        rb_info = rpc.get_worker_info("replay_buffer")
         

        parameter_server_ref:rpc.RRef = remote(ps_info, get_remote_parameter_server)
        replay_buffer_ref:rpc.RRef = remote(rb_info, get_remote_replay_buffer)
        learner = Learner(parameter_server_ref,replay_buffer_ref,
                            prior_eps,
                            gamma,
                            alpha,
                            beta,
                            tau,
                            lr)
        
        print("learner while loop start")
        try:
            learn_count = 0
            while True:
                # 检查是否有足够的经验进行学习
                buffer_size = replay_buffer_ref.rpc_sync().buffer_size()                 
                should_learn = parameter_server_ref.rpc_sync().should_start_learning()            

                learner.logger.info(f"buffer_size: {buffer_size} , should_learn: {should_learn}")
                if should_learn:
                    if buffer_size >= learner.batch_size+1: 
                        learner.train_step()
                        learn_count  +=1
                        if learn_count % 1000 == 1:
                            parameter_server_ref.rpc_sync().push_eval_params()
                            learner.logger.info(f"触发评估 eval_count: {learn_count}")
                        
                    else:
                        parameter_server_ref.rpc_sync().reset_batch_completion()  
                        parameter_server_ref.rpc_sync().set_learning_status(False)                        
                else:
                    learner.logger.info(f"ReplayBuffer大小({buffer_size})小于batch_size({learner.batch_size+1})，等待更多经验")
                    # 等待一段时间再检查
                    
                    time.sleep(3)
        except KeyboardInterrupt:
            learner.logger.info("收到中断信号，Learner停止运行")
        except Exception as e:
            learner.logger.error(f"Learner运行出错: {e}", exc_info=True)
        finally:
            rpc.shutdown()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.0.109", help="host for distributed training")
    parser.add_argument("--world_size", type=int, default=5, help="world size for distributed training")    
    parser.add_argument("--lr", type=float, default=1e-4)  # 学习率 / Learning rate
    parser.add_argument("--gamma", type=float, default=0.99)  # 折扣因子 / Discount factor
    parser.add_argument("--tau", type=float, default=0.005)  # 软更新系数 / Soft update coefficient
    parser.add_argument("--alpha",type=float,default=0.5)  # KL损失权重系数 / KL loss weight coefficient #alpha 代表了记忆的经验权重， beta代表先验q-table权重
    parser.add_argument("--beta",type=int,default=5) #alpha 代表了记忆的经验权重， beta代表先验q-table权重
    parser.add_argument("--prior_eps", type=float, default=1e-6)  # 优先级epsilon / Prior epsilon
    
    
    args = parser.parse_args()
    
    world_size = args.world_size
    host = args.host    
    Learner.run(2, host, world_size,
                args.prior_eps,
                args.gamma,
                args.alpha,
                args.beta,
                args.tau,
                args.lr
                ) 