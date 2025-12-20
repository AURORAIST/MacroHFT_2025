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
from RL.agent.rpc.config import action_dim,seed,tech_indicator_list,tech_indicator_list_trend,n_state_1_dim,n_state_2_dim
from env.high_level_env import Testing_Env
import numpy as np
from model.net import *
from RL.util.logging_config import setup_logger
from RL.agent.rpc.parameter_server import get_remote_parameter_server

device = "cpu"

class Evaluator:
    def __init__(self, 
                 dataset,   
                 transcation_cost,
                 back_time_length,
                 exp
                 ):
        self.logger = setup_logger("evaluator", "distributed_dqn.log", role="Evaluator")
        self.logger.info(f"Evaluator初始化，state_dim: {n_state_1_dim},{n_state_2_dim} action_dim: {action_dim}") 

        self.val_data_path = os.path.join(ROOT, "data", dataset, "whole")
        self.device = device
        self.tech_indicator_list=tech_indicator_list
        self.tech_indicator_list_trend=tech_indicator_list_trend
        self.transcation_cost=transcation_cost
        self.back_time_length=back_time_length
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
        self.df = pd.read_feather(os.path.join(self.val_data_path, "val.feather")).head(100)
        
        self.n_action = action_dim
        self.n_state_1 = n_state_1_dim
        self.n_state_2 = n_state_2_dim
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
              
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(device)
        self.hyperagent.eval()
        self.clf_list = ['slope_360', 'vol_360'] 
        self.exp = exp
        self.result_path = os.path.join("./result/high_level", '{}'.format(self.dataset), exp)
        self.best_return_rate = 0

    def act_test(self, state, state_trend, state_clf, info):
        """
        测试模式下的动作选择（无随机性）
        
        Args:
            state: 当前状态
            state_trend: 状态趋势
            state_clf: 状态分类特征
            info: 包含历史动作等信息的字典
            
        Returns:
            int: 确定性选择的最优动作
        """
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0).to(self.device)
            qs = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action
        
    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)# qs将6个代理的2个动作的权重 [6，2]  => [6,1,2]
        q_tensor = q_tensor.permute(1, 0, 2) # 重排维度为 [1,6,2]
        weights_reshaped = w.view(-1, 1, 6) #  超代理评估的6个子代理权重为[1,6], 改变形状为[1，1，6]
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1) # 执行批量矩阵乘法[1,1,2]并压缩维度得到最终Q值 [1, 2]
        
        return combined_q
             
    def val_cluster(self,epoch_counter,params):
        for p, new_p in zip(self.hyperagent.parameters(), params):
            p.data.copy_(new_p)
        
        
        epoch_path = os.path.join(self.result_path, "epoch_{}".format(epoch_counter))
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        trained_model_file =  os.path.join(epoch_path, "trained_model.pkl")
        torch.save(self.hyperagent.state_dict(), trained_model_file)  
        self.logger.info(f"保存模型 {epoch_counter} {trained_model_file}")
        save_path = os.path.join(epoch_path, "val")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []        
        
        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,               
                clf_list=self.clf_list,    
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = val_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = val_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        return_margin, pure_balance, required_money, commission_fee = val_env.get_final_return_rate(slient=True)
        final_balance = pure_balance + val_env.calculate_value(info_['previous_price_information'], val_env.position)
        portfit_margine = final_balance / required_money
        self.logger.info(f"val return_margin:{return_margin:.2f},portfit_margine:{portfit_margine:.2f},final_balance:{final_balance:.2f},pure_balance:{pure_balance:.2f},required_money:{required_money:.2f},commission_fee:{commission_fee:.2f}")
     

        
        final_balance = val_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action_val.npy"), action_list)
        np.save(os.path.join(save_path, "reward_val.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance_val.npy"), final_balance_list)
        np.save(os.path.join(save_path, "require_money_val.npy"), required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val.npy"), commission_fee_list)
        return_rate = final_balance / required_money
        
        
        
        
        return return_rate
    
    
    def save_best(self,return_rate_eval,params): 
        if return_rate_eval > self.best_return_rate:
            self.best_return_rate = return_rate_eval           
            best_model_path = os.path.join("./result/high_level", '{}'.format(self.dataset), self.exp, 'best_model.pkl')
            torch.save(params, best_model_path)
            self.logger.info(f"更新最佳模型 {return_rate_eval}")

    @staticmethod
    def run(rank, host, world_size ,               
                dataset,                 
                transcation_cost,
                back_time_length,
                exp
                 ):
        print("Evaluator run start")
   
        rpc.init_rpc(
            f"Evaluator_{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{host}:29500"  # Parameter Server运行在machine1上
            )
        )
        print("Evaluator init")
        ps_info = rpc.get_worker_info("parameter_server") 
        parameter_server_ref:rpc.RRef = remote(ps_info, get_remote_parameter_server) 
        evaluator = Evaluator(  dataset,   
                                transcation_cost,
                                back_time_length,
                                exp
                            )
        
        try:
            while True:
                # 等待学习完成
                evaluator.logger.info("等待需要评估的参数")
                epoch_counter,params = parameter_server_ref.rpc_sync().get_eval_params()
                if epoch_counter is None:
                    time.sleep(50)
                else:
                    evaluator.logger.info(f"开始评估 {epoch_counter}")
                    return_rate_eval = evaluator.val_cluster(epoch_counter,params)
                    evaluator.save_best(return_rate_eval,params)
            pass
        except KeyboardInterrupt:
            evaluator.logger.info("收到中断信号，Evaluator停止运行")
        except Exception as e:
            evaluator.logger.error(f"Evaluator运行出错: {e}", exc_info=True)

        finally:
            rpc.shutdown()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.0.109", help="host for distributed training")
    parser.add_argument("--node_rank", type=int, default=1, help="Node rank for distributed training")
    parser.add_argument("--world_size", type=int, default=5, help="world size for distributed training")     
    parser.add_argument("--dataset", type=str, default="ETHUSDT")  # 数据集名称 / Dataset name
    parser.add_argument("--transcation_cost", type=float, default=2.0 / 100000)  # 交易成本 / Transaction cost
    parser.add_argument("--back_time_length", type=int, default=1)  # 回滚时间长度 / Rollback time length
    parser.add_argument("--exp",type=str,default="exp1")
    
    args = parser.parse_args()
    
    world_size = args.world_size
    host = args.host   
    Evaluator.run(3, host, world_size,              
                args.dataset,             
                args.transcation_cost,
                args.back_time_length,
                args.exp
               )