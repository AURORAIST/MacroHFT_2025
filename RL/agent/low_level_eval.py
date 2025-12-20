import logging as log
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from env.low_level_env import Testing_Env, Training_Env
import numpy as np
from model.net import *
import multiprocessing

# Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization issues
multiprocessing.set_start_method('spawn', force=True)

# 辅助函数，用于在子进程中执行验证任务
def _validate_worker(args):
    """
    在子进程中执行验证任务的辅助函数
    
    中文说明：
    此函数被设计为在子进程中运行，接收验证所需的所有参数，
    创建EVALER实例并执行验证，返回验证结果。
    
    English description:
    This function is designed to run in a child process, receives all parameters needed for validation,
    creates an EVALER instance and performs validation, returns validation results.
    
    Parameters:
        args (tuple): 包含验证所需的所有参数的元组
            - n_state_1: 状态空间维度1
            - n_state_2: 状态空间维度2
            - n_action: 动作空间维度
            - device: 计算设备
            - val_data_path: 验证数据路径
            - tech_indicator_list: 技术指标列表
            - tech_indicator_list_trend: 趋势技术指标列表
            - transcation_cost: 交易成本
            - back_time_length: 回溯时间长度
            - max_holding_number: 最大持仓数量
            - epoch_path: 模型路径
            - df_id: 数据文件ID
            - initial_action: 初始动作
            
    Returns:
        tuple: 验证结果 (action_list_episode, reward_list_episode, final_balance, required_money, commission_fee)
    """
    # 解包参数
    (n_state_1, n_state_2, n_action, device, val_data_path, 
     tech_indicator_list, tech_indicator_list_trend, transcation_cost,
     back_time_length, max_holding_number, epoch_path, df_id, initial_action) = args
    
    # 创建EVALER实例
    evaler = EVALER(n_state_1, n_state_2, n_action, device,
                   val_data_path, tech_indicator_list, tech_indicator_list_trend,
                   transcation_cost, back_time_length, max_holding_number)
    
    # 执行验证并返回结果
    return evaler.val_cluster(epoch_path, df_id, initial_action)

class EVALER(object):
    def __init__(self, n_state_1,n_state_2,n_action,device,
                 val_data_path,
                 tech_indicator_list,
                 tech_indicator_list_trend,
                 transcation_cost,
                 back_time_length,
                 max_holding_number,
                 ):  
        self.device = torch.device(device) 
        self.val_data_path = val_data_path
        self.eval_net = subagent(n_state_1, n_state_2, n_action, 128).to(self.device)

        self.tech_indicator_list=tech_indicator_list
        self.tech_indicator_list_trend=tech_indicator_list_trend
        self.transcation_cost=transcation_cost
        self.back_time_length=back_time_length
        self.max_holding_number=max_holding_number
        
    def test_select_action(self, state, state_trend, info):
        """
        测试阶段选择最优动作（无随机探索）
        
        中文说明：
        在测试阶段使用确定性策略选择最优动作，
        不进行随机探索，完全依赖网络预测。
        
        English description:
        Selects optimal action deterministically during testing phase,
        no random exploration, fully relies on network prediction.
        
        Parameters:
            state (np.array): 当前状态 / Current state
            state_trend (np.array): 当前趋势状态 / Current trend state
            info (dict): 包含历史信息的字典 / Dictionary containing historical information
            
        Returns:
            int: 选择的动作编号 / Selected action number
        """
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        actions_value = self.eval_net(x1, x2, previous_action)
        # 选择最优动作 / Select optimal action
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action
    
    def val_cluster(self, epoch_path, df_id, initial_action):
        # 加载训练模型 / Load trained model
        loaded_params = torch.load(os.path.join(epoch_path, "trained_model.pkl"), map_location=self.device)
        self.eval_net.load_state_dict(loaded_params)
        self.eval_net.to(self.device)
        # 设置为验证模式 / Set evaluation mode
        self.eval_net.eval()
        log.info(f"validating on df {df_id}")
        # 加载验证数据文件 / Load validation data file
        self.df = pd.read_feather(os.path.join(self.val_data_path, "df_{}.feather".format(df_id)))
        # .head(100)
        # 初始化测试环境 / Initialize testing environment         
        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=initial_action)
        # 重置环境获取初始状态 / Reset environment to get initial state
        single_state, trend_state, info = val_env.reset()
        done = False
        # 单次验证过程的临时存储 / Temporary storage for current validation episode
        action_list_episode = []
        reward_list_episode = []
        # 执行验证交互循环 / Execute validation interaction loop
        while not done:
            # 选择测试动作 / Select test action
            action = self.test_select_action(single_state, trend_state, info)
            # 执行动作获取下一个状态 / Execute action to get next state
            next_single_state, next_trend_state, reward, done, next_info = val_env.step(action)
            # 收集奖励和状态信息 / Collect reward and state information
            reward_list_episode.append(reward)
            single_state, trend_state, info = next_single_state, next_trend_state, next_info
            action_list_episode.append(action)
            # 获取账户信息（静默模式） / Get account information (silent mode)
            portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(slient=True)
        # 获取最终账户信息 / Get final account information
        final_balance = val_env.final_balance
        required_money = val_env.required_money       
        return action_list_episode, reward_list_episode ,final_balance, required_money, commission_fee
    
class DQN_EVAL(object):
    def __init__(self, n_state_1,n_state_2,n_action,device,
                 val_data_path,
                 tech_indicator_list,
                 tech_indicator_list_trend,
                 transcation_cost,
                 back_time_length,
                 max_holding_number,
                 ):  
        self.n_state_1= n_state_1
        self.n_state_2= n_state_2
        self.n_action= n_action
        self.val_data_path = val_data_path
        # self.eval_net = subagent(n_state_1, n_state_2, n_action, 64).to(device)
        self.device = device
        self.tech_indicator_list=tech_indicator_list
        self.tech_indicator_list_trend=tech_indicator_list_trend
        self.transcation_cost=transcation_cost
        self.back_time_length=back_time_length
        self.max_holding_number=max_holding_number
    

    
    def _save_val_result(self, 
                        save_path, 
                        initial_action,
                        action_list,
                        reward_list,
                        final_balance_list,
                        required_money_list,
                        commission_fee_list):
        """
        保存验证结果数据并计算平均收益率
        
        中文说明：
        本函数负责将验证过程中收集的动作、奖励、账户余额等数据
        转换为numpy数组并保存到指定路径，同时计算并返回平均收益率。
        
        English description:
        This function converts validation data including actions, rewards, account balances
        into numpy arrays and saves them to specified path. It also calculates and returns
        the mean return rate.
        
        Parameters:
            save_path (str): 结果保存路径 / Path to save validation results
            initial_action (int): 初始动作标识 / Initial action identifier
            action_list (list): 动作序列列表 / List of action sequences
            reward_list (list): 奖励列表 / List of rewards
            final_balance_list (list): 最终余额列表 / List of final balances
            required_money_list (list): 所需资金列表 / List of required money
            commission_fee_list (list): 手续费列表 / List of commission fees
            
        Returns:
            float: 计算得到的平均收益率（已处理NaN值）
        """
        # 转换数据为numpy数组 / Convert lists to numpy arrays
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        # 保存验证数据到npy文件 / Save validation data to npy files
        np.save(os.path.join(save_path, "action_val_{}.npy".format(initial_action)), action_list)
        np.save(os.path.join(save_path, "reward_val_{}.npy".format(initial_action)), reward_list)
        np.save(os.path.join(save_path, "final_balance_val_{}.npy".format(initial_action)), final_balance_list)
        np.save(os.path.join(save_path, "require_money_val_{}.npy".format(initial_action)), required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val_{}.npy".format(initial_action)), commission_fee_list)
        # 计算平均收益率并保存 / Calculate and save mean return rate
        # 使用nan_to_num处理可能存在的NaN值 / Handle possible NaN values with nan_to_num
        return_rate_mean = np.nan_to_num(final_balance_list / required_money_list).mean()
        np.save(os.path.join(save_path, "return_rate_mean_val_{}.npy".format(initial_action)), return_rate_mean)
        # 返回计算结果 / Return calculated mean return rate
        return return_rate_mean    
        
        
    def val_cluster(self, epoch_path, save_path, initial_action, df_list):
        """
        在验证集上评估模型性能并生成评估结果
        
        中文说明：
        本函数负责加载指定周期的训练模型，在验证集上执行测试，
        收集动作、奖励、账户余额等指标，并保存验证结果。
        使用多进程并行处理验证任务以提高效率。
        
        English description:
        This function loads the trained model for specified epoch, executes testing on validation set,
        collects metrics including actions, rewards, account balances, and saves validation results.
        Uses multiprocessing to parallelize validation tasks for improved efficiency.
        
        Parameters:
            epoch_path (str): 模型所在目录路径 / Path to trained model directory
            save_path (str): 结果保存目录路径 / Path to save validation results
            initial_action (int): 初始动作标识 / Initial action identifier
            df_list (list): 数据文件ID列表 / List of data file IDs
            
        Returns:
            float: 计算得到的平均收益率（已处理NaN值）
        """
        
        # 获取验证数据索引列表 / Get validation data index list
        # df_number = int(len(df_list)) 
        # 初始化验证数据收集容器 / Initialize containers for validation data collection
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        
        # 准备多进程参数 / Prepare multiprocessing parameters
        # 获取CPU核心数，但限制最大进程数以避免资源耗尽
        num_processes = 5
        log.info(f"Using {num_processes} processes for parallel validation df_list:{df_list}")
        
        # 创建参数列表，每个元素是一个包含所有必要参数的元组
        args_list = [
            (self.n_state_1, self.n_state_2, self.n_action, self.device,
             self.val_data_path, self.tech_indicator_list, self.tech_indicator_list_trend,
             self.transcation_cost, self.back_time_length, self.max_holding_number,
             epoch_path, df_id, initial_action)
            for df_id in df_list
        ]
        
        # 创建进程池并并行执行验证任务
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用imap_unordered以获取结果顺序无关的方式提高效率
            results = pool.imap_unordered(_validate_worker, args_list)
            
            # 收集结果
            for result in results:
                action_list_episode, reward_list_episode, final_balance, required_money, commission_fee = result
                log.info(f"val _validate_worker  final_balance: {final_balance}, required_money: {required_money}, commission_fee: {commission_fee}")
                action_list.append(action_list_episode)
                reward_list.append(reward_list_episode)
                final_balance_list.append(final_balance)
                required_money_list.append(required_money)
                commission_fee_list.append(commission_fee)
        
        # 保存验证结果并计算平均收益率 / Save validation results and calculate mean return rate
        return_rate_mean = self._save_val_result(save_path, initial_action, action_list, reward_list, final_balance_list, required_money_list, commission_fee_list)
        log.info(f"val return_rate_mean:{return_rate_mean}")
        return return_rate_mean