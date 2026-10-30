#!/usr/bin/env python
"""
使用预训练模型进行评估的脚本
不需要训练，直接加载模型进行验证和测试
"""

import pathlib
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
import os
from pathlib import Path
import logging
from datetime import datetime

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from model.net import *
from env.low_level_env import Testing_Env
from RL.util.utili import LinearDecaySchedule

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser(description='Evaluate pretrained low-level model')
parser.add_argument("--pretrained_model", type=str, required=True, 
                    help="Path to pretrained model (.pkl file)")
parser.add_argument("--dataset", type=str, default="ETHUSDT")
parser.add_argument("--clf", type=str, default="slope", choices=["slope", "vol"])
parser.add_argument("--label", type=str, default="label_1")
parser.add_argument("--num_actions", type=int, default=2)
parser.add_argument("--transcation_cost", type=float, default=2.0 / 10000)
parser.add_argument("--back_time_length", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--eval_set", type=str, default="val", 
                    choices=["val", "test"], help="Which dataset to evaluate on")
parser.add_argument("--save_path", type=str, default=None,
                    help="Path to save evaluation results (default: auto-generate)")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ModelEvaluator:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        seed_torch(self.seed)
        
        # 设置设备
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        
        # 设置路径
        self.dataset = args.dataset
        self.clf = args.clf
        self.label = int(args.label.split('_')[1])
        
        # 数据路径
        if args.eval_set == "val":
            self.data_path = os.path.join(ROOT, "data", args.dataset, "val")
        else:
            self.data_path = os.path.join(ROOT, "data", args.dataset, "test")
        
        # 加载索引
        if args.clf == 'slope':
            with open(os.path.join(self.data_path, 'slope_labels.pkl'), 'rb') as file:
                self.data_index = pickle.load(file)
        elif args.clf == 'vol':
            with open(os.path.join(self.data_path, 'vol_labels.pkl'), 'rb') as file:
                self.data_index = pickle.load(file)
        
        # 设置max_holding_number
        if "BTC" in self.dataset:
            self.max_holding_number = 0.01
        elif "ETH" in self.dataset:
            self.max_holding_number = 0.2
        elif "DOT" in self.dataset:
            self.max_holding_number = 10
        elif "LTC" in self.dataset:
            self.max_holding_number = 10
        else:
            raise Exception("Unsupported dataset")
        
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = args.num_actions
        
        # 加载特征列表
        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', 
                                           allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', 
                                                 allow_pickle=True).tolist()
        
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        
        # 创建网络
        self.eval_net = subagent(self.n_state_1, self.n_state_2, 
                                self.n_action, 64).to(self.device)
        
        # 设置保存路径（在加载模型前需要获取model_name）
        if args.save_path is None:
            model_name = Path(args.pretrained_model).stem
            param_str = f"clf{args.clf}_label{args.label}_act{args.num_actions}_cost{args.transcation_cost}_back{args.back_time_length}_seed{args.seed}"
            self.save_path = f"./result/low_level/{args.dataset}/eval_{model_name}_{args.eval_set}_{param_str}"
        else:
            self.save_path = args.save_path
            model_name = Path(args.pretrained_model).stem
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # 设置日志（必须在load_model之前初始化）
        log_dir = f"./logs/low_level/{args.dataset}"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"eval_{model_name}_{args.eval_set}_{timestamp}.log")
        
        # 配置日志格式 - 同时输出到文件和控制台
        # 清除现有的handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',  # 简化格式，只显示消息
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8', mode='w'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
        # 将print重定向到logger
        class LoggerWriter:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level
                self.linebuf = ''
            
            def write(self, message):
                if message != '\n':
                    self.logger.log(self.level, message.rstrip())
            
            def flush(self):
                pass
        
        # 重定向stdout到logger
        sys.stdout = LoggerWriter(self.logger, logging.INFO)
        sys.stderr = LoggerWriter(self.logger, logging.ERROR)
        
        # 加载预训练模型（在logger初始化之后）
        self.load_model(args.pretrained_model)
        self.eval_net.eval()
        
        print("\n" + "="*80)
        print("Model Evaluator Initialized".center(80))
        print("="*80)
        print(f"Dataset: {args.dataset}")
        print(f"Classifier: {args.clf}")
        print(f"Label: {args.label}")
        print(f"Eval Set: {args.eval_set}")
        print(f"Num Actions: {args.num_actions}")
        print(f"Save Path: {self.save_path}")
        print(f"Log File: {log_file}")
        print("="*80 + "\n")
        
        self.logger.info("="*80)
        self.logger.info("Model Evaluator Initialized")
        self.logger.info("="*80)
        self.logger.info(f"Dataset: {args.dataset}")
        self.logger.info(f"Classifier: {args.clf}")
        self.logger.info(f"Label: {args.label}")
        self.logger.info(f"Eval Set: {args.eval_set}")
        self.logger.info(f"Num Actions: {args.num_actions}")
        self.logger.info(f"Model Path: {args.pretrained_model}")
        self.logger.info(f"Save Path: {self.save_path}")
        self.logger.info(f"Log File: {log_file}")
        self.logger.info("="*80)
    
    def load_model(self, model_path):
        """加载预训练模型"""
        print(f"\nLoading model from: {model_path}")
        self.logger.info(f"Loading model from: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.eval_net.load_state_dict(state_dict)
            print("✓ Model loaded successfully!\n")
            self.logger.info("✓ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.logger.error(f"❌ Error loading model: {e}")
            raise
    
    def act_test(self, state, state_trend, info):
        """测试时的动作选择（贪婪策略）"""
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0).to(self.device)
        
        with torch.no_grad():
            actions_value = self.eval_net(x1, x2, previous_action)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        
        return action
    
    def evaluate(self):
        """在验证集或测试集上评估"""
        df_list = self.data_index[self.label]
        df_number = int(len(df_list))
        
        print(f"\nEvaluating on {df_number} dataframes...")
        print(f"Testing all {self.n_action} initial positions...\n")
        
        self.logger.info(f"Starting evaluation on {df_number} dataframes")
        self.logger.info(f"Testing all {self.n_action} initial positions")
        
        all_results = []
        
        # 对每个初始仓位进行评估
        for init_action in range(self.n_action):
            print(f"\n{'='*80}")
            print(f"Initial Position {init_action} ({init_action/(self.n_action-1)*100:.0f}%)".center(80))
            print("="*80)
            
            self.logger.info("="*80)
            self.logger.info(f"Initial Position {init_action} ({init_action/(self.n_action-1)*100:.0f}%)")
            self.logger.info("="*80)
            
            action_list = []
            reward_list = []
            final_balance_list = []
            required_money_list = []
            commission_fee_list = []
            
            for i in range(df_number):
                df_index = df_list[i]
                print(f"Evaluating on df {df_index}... ", end="", flush=True)
                
                df = pd.read_feather(
                    os.path.join(self.data_path, f"df_{df_index}.feather"))
                
                eval_env = Testing_Env(
                    df=df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=init_action,
                    num_actions=self.n_action
                )
                
                s, s2, info = eval_env.reset()
                done = False
                action_list_episode = []
                reward_list_episode = []
                
                while not done:
                    a = self.act_test(s, s2, info)
                    s_, s2_, r, done, info_ = eval_env.step(a)
                    reward_list_episode.append(r)
                    s, s2, info = s_, s2_, info_
                    action_list_episode.append(a)
                
                portfit_margin, final_balance, required_money, commission_fee = \
                    eval_env.get_final_return_rate(slient=True)
                
                final_balance = eval_env.final_balance
                required_money = eval_env.required_money
                
                action_list.append(action_list_episode)
                reward_list.append(reward_list_episode)
                final_balance_list.append(final_balance)
                required_money_list.append(required_money)
                commission_fee_list.append(commission_fee)
                
                return_rate = (final_balance / required_money) * 100
                print(f"Return: {return_rate:.2f}%")
            
            # 转换为numpy数组
            action_list = np.array(action_list, dtype=object)
            reward_list = np.array(reward_list, dtype=object)
            final_balance_list = np.array(final_balance_list)
            required_money_list = np.array(required_money_list)
            commission_fee_list = np.array(commission_fee_list)
            
            # 计算平均收益率
            return_rate_mean = np.nan_to_num(
                final_balance_list / required_money_list).mean()
            
            # 统计动作分布
            action_counts = np.zeros(self.n_action)
            total_actions = 0
            for episode_actions in action_list:
                for a in episode_actions:
                    action_counts[int(a)] += 1
                    total_actions += 1
            
            action_distribution = action_counts / total_actions if total_actions > 0 else action_counts
            
            # 打印统计信息
            print(f"\n--- Evaluation Statistics (Initial Position: {init_action}) ---")
            print(f"Total steps: {total_actions}")
            print("Action distribution:")
            scale_factor = self.n_action - 1
            for i in range(self.n_action):
                position_pct = (i / scale_factor if scale_factor > 0 else 0) * 100
                count = int(action_counts[i])
                pct = action_distribution[i] * 100
                print(f"  Action {i} ({position_pct:5.1f}% position): "
                      f"{count:6d} times ({pct:5.2f}%)")
            print(f"Mean return rate: {return_rate_mean:.4f} ({return_rate_mean*100:.2f}%)")
            print("-" * 60)
            
            # 记录日志
            self.logger.info(f"--- Evaluation Statistics (Initial Position: {init_action}) ---")
            self.logger.info(f"Total steps: {total_actions}")
            self.logger.info("Action distribution:")
            for i in range(self.n_action):
                position_pct = (i / scale_factor if scale_factor > 0 else 0) * 100
                count = int(action_counts[i])
                pct = action_distribution[i] * 100
                self.logger.info(f"  Action {i} ({position_pct:5.1f}% position): "
                                f"{count:6d} times ({pct:5.2f}%)")
            self.logger.info(f"Mean return rate: {return_rate_mean:.4f} ({return_rate_mean*100:.2f}%)")
            self.logger.info("-" * 60)
            
            # 保存结果
            save_path_init = os.path.join(self.save_path, f"init_pos_{init_action}")
            os.makedirs(save_path_init, exist_ok=True)
            
            np.save(os.path.join(save_path_init, "action.npy"), action_list)
            np.save(os.path.join(save_path_init, "reward.npy"), reward_list)
            np.save(os.path.join(save_path_init, "final_balance.npy"), final_balance_list)
            np.save(os.path.join(save_path_init, "require_money.npy"), required_money_list)
            np.save(os.path.join(save_path_init, "commission_fee_history.npy"), commission_fee_list)
            np.save(os.path.join(save_path_init, "action_distribution.npy"), action_distribution)
            np.save(os.path.join(save_path_init, "return_rate_mean.npy"), return_rate_mean)
            
            all_results.append({
                'init_action': init_action,
                'return_rate_mean': return_rate_mean,
                'action_distribution': action_distribution.tolist()
            })
        
        # 打印总结
        print("\n" + "="*80)
        print("Evaluation Summary".center(80))
        print("="*80)
        for result in all_results:
            print(f"Initial Position {result['init_action']}: "
                  f"{result['return_rate_mean']*100:.2f}%")
        
        avg_return = np.mean([r['return_rate_mean'] for r in all_results])
        print(f"\nAverage Return Rate: {avg_return*100:.2f}%")
        print(f"\nResults saved to: {self.save_path}")
        print("="*80 + "\n")
        
        # 记录日志总结
        self.logger.info("="*80)
        self.logger.info("Evaluation Summary")
        self.logger.info("="*80)
        for result in all_results:
            self.logger.info(f"Initial Position {result['init_action']}: "
                            f"{result['return_rate_mean']*100:.2f}%")
        self.logger.info(f"\nAverage Return Rate: {avg_return*100:.2f}%")
        self.logger.info(f"Results saved to: {self.save_path}")
        self.logger.info("="*80)
        
        # 保存总结
        import json
        with open(os.path.join(self.save_path, "summary.json"), 'w') as f:
            json.dump({
                'model_path': self.args.pretrained_model,
                'dataset': self.args.dataset,
                'eval_set': self.args.eval_set,
                'clf': self.args.clf,
                'label': self.args.label,
                'num_actions': self.n_action,
                'results': all_results,
                'average_return_rate': float(avg_return)
            }, f, indent=2)
        
        return all_results


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    
    evaluator = ModelEvaluator(args)
    results = evaluator.evaluate()
    
    print("\n✓ Evaluation completed!")
