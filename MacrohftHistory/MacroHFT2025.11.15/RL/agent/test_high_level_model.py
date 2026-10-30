#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
独立测试脚本 - 用于单独测试已训练的高层策略模型
可以在训练完成后单独运行，或从上一次中断处恢复测试

使用方法：
    python RL/agent/test_high_level_model.py \
        --model-path ./result/high_level/ETHUSDT/exp1_k210epo/pos_2/seed_12345/emergency_save_epoch_1/current_model.pkl \
        --dataset ETHUSDT \
        --device cuda:0 \
        --num_actions 2 \
        --output-dir ./result/high_level/ETHUSDT/exp1/pos_3/seed_12345/exp1_k210epo
"""

import pathlib
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from model.net import *
from env.high_level_env import Testing_Env
from RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser(
    description='独立测试高层策略模型'
)
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="模型文件路径 (e.g., ./result/high_level/ETHUSDT/exp1/pos_2/seed_12345/best_model.pkl)"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="ETHUSDT",
    help="数据集名称"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="设备 (cuda:0, cpu 等)"
)
parser.add_argument(
    "--num_actions",
    type=int,
    default=2,
    help="高层策略的动作数量"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="输出目录 (默认: model_path 的上级目录/test)"
)
parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0.2 / 1000,
    help="交易成本"
)
parser.add_argument(
    "--back_time_length",
    type=int,
    default=1,
    help="回溯时间长度"
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="随机种子"
)
parser.add_argument(
    "--log-dir",
    type=str,
    default=None,
    help="TensorBoard 日志目录 (可选)"
)


def seed_torch(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HighLevelTester:
    """高层策略测试类"""
    
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        
        # 设备设置
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
            print(f"⚠️  CUDA not available, using CPU instead")
        
        # 数据路径
        self.dataset = args.dataset
        self.test_data_path = os.path.join(ROOT, "data", args.dataset, "whole")
        
        # 输出路径
        if args.output_dir:
            self.output_dir = args.output_dir
        else:
            self.output_dir = os.path.join(
                os.path.dirname(args.model_path),
                "test_standalone"
            )
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print(f"Output directory: {self.output_dir}")
        
        # TensorBoard 日志
        if args.log_dir:
            self.log_dir = args.log_dir
        else:
            self.log_dir = os.path.join(self.output_dir, "log")
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.writer = SummaryWriter(self.log_dir)
        
        # 参数
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = args.num_actions
        
        # 数据集特定参数
        if "BTC" in self.dataset:
            self.max_holding_number = 0.01
        elif "ETH" in self.dataset:
            self.max_holding_number = 0.2
        elif "DOT" in self.dataset:
            self.max_holding_number = 10
        elif "LTC" in self.dataset:
            self.max_holding_number = 10
        else:
            raise Exception(f"Unsupported dataset: {self.dataset}")
        
        # 加载特征列表
        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
        self.clf_list = ['slope_360', 'vol_360']
        
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        
        # 初始化模型
        print(f"\nInitializing models...")
        self._init_sub_agents()
        self._init_hyperagent(args.model_path)
    
    def _init_sub_agents(self):
        """初始化6个子代理（低层模型）"""
        print(f"Loading sub-agents...")
        
        self.slope_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_1 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_2 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_3 = subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        
        # 加载预训练的低层模型
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
        
        try:
            self.slope_1.load_state_dict(torch.load(model_list_slope[0], map_location=self.device))
            self.slope_2.load_state_dict(torch.load(model_list_slope[1], map_location=self.device))
            self.slope_3.load_state_dict(torch.load(model_list_slope[2], map_location=self.device))
            self.vol_1.load_state_dict(torch.load(model_list_vol[0], map_location=self.device))
            self.vol_2.load_state_dict(torch.load(model_list_vol[1], map_location=self.device))
            self.vol_3.load_state_dict(torch.load(model_list_vol[2], map_location=self.device))
            print(f"✅ Sub-agents loaded successfully")
        except FileNotFoundError as e:
            print(f"⚠️  Sub-agent file not found: {e}")
            print(f"   Creating random initialized sub-agents instead")
        
        self.slope_1.eval()
        self.slope_2.eval()
        self.slope_3.eval()
        self.vol_1.eval()
        self.vol_2.eval()
        self.vol_3.eval()
        
        self.slope_agents = {0: self.slope_1, 1: self.slope_2, 2: self.slope_3}
        self.vol_agents = {0: self.vol_1, 1: self.vol_2, 2: self.vol_3}
    
    def _init_hyperagent(self, model_path):
        """初始化高层代理并加载模型"""
        print(f"Loading hyperagent from: {model_path}")
        
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.hyperagent.load_state_dict(state_dict)
            print(f"✅ Hyperagent loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load hyperagent: {e}")
        
        self.hyperagent.eval()
    
    def calculate_q(self, w, qs):
        """计算加权 Q 值"""
        q_tensor = torch.stack(qs)
        q_tensor = q_tensor.permute(1, 0, 2)
        weights_reshaped = w.view(-1, 1, 6)
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
        return combined_q
    
    def act_test(self, state, state_trend, state_clf, info):
        """测试动作选择"""
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor(info["previous_action"]).long().to(self.device),
                0).to(self.device)
            
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
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            return action
    
    def test(self):
        """运行测试"""
        print(f"\n{'='*80}")
        print(f"Starting Test")
        print(f"{'='*80}")
        print(f"Dataset: {self.dataset}")
        print(f"Number of actions: {self.n_action}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        try:
            # 加载测试数据
            print(f"Loading test data from: {self.test_data_path}")
            df = pd.read_feather(os.path.join(self.test_data_path, "test.feather"))
            print(f"✅ Test data loaded: {len(df)} rows")
            sys.stdout.flush()
            
            # 初始化测试环境
            print(f"Initializing test environment...")
            test_env = Testing_Env(
                df=df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0,
                num_actions=self.n_action
            )
            print(f"✅ Test environment initialized")
            sys.stdout.flush()
            
            # 重置环境
            print(f"Resetting environment...")
            s, s2, s3, info = test_env.reset()
            print(f"✅ Environment reset")
            sys.stdout.flush()
            
            # 运行测试
            print(f"\nRunning test episode...")
            print(f"  Total steps: (calculating...)")
            sys.stdout.flush()
            
            done = False
            action_list_episode = []
            reward_list_episode = []
            step_count = 0
            
            while not done:
                try:
                    a = self.act_test(s, s2, s3, info)
                    s_, s2_, s3_, r, done, info_ = test_env.step(a)
                    reward_list_episode.append(r)
                    s, s2, s3, info = s_, s2_, s3_, info_
                    action_list_episode.append(a)
                    step_count += 1
                    
                    # 定期输出进度
                    if step_count % 1000 == 0:
                        print(f"  Step {step_count} completed...")
                        sys.stdout.flush()
                    
                except Exception as e:
                    print(f"❌ Error during step {step_count}: {e}")
                    print(f"   Attempting to recover...")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    raise
            
            print(f"✅ Episode completed: {step_count} steps")
            sys.stdout.flush()
            
            # 获取最终结果
            print(f"Calculating final metrics...")
            portfit_magine, final_balance, required_money, commission_fee = test_env.get_final_return_rate(slient=True)
            final_balance = test_env.final_balance
            
            # 保存结果
            print(f"Saving results...")
            action_list = np.array([action_list_episode])
            reward_list = np.array([reward_list_episode])
            final_balance_list = np.array([final_balance])
            required_money_list = np.array([required_money])
            commission_fee_list = np.array([commission_fee])
            
            np.save(os.path.join(self.output_dir, "action.npy"), action_list)
            np.save(os.path.join(self.output_dir, "reward.npy"), reward_list)
            np.save(os.path.join(self.output_dir, "final_balance.npy"), final_balance_list)
            np.save(os.path.join(self.output_dir, "require_money.npy"), required_money_list)
            np.save(os.path.join(self.output_dir, "commission_fee_history.npy"), commission_fee_list)
            print(f"✅ Results saved to: {self.output_dir}")
            sys.stdout.flush()
            
            # 计算关键指标
            return_rate = final_balance / required_money
            reward_sum = np.sum(reward_list)
            
            # 输出到 TensorBoard
            self.writer.add_scalar(
                tag="test/return_rate",
                scalar_value=return_rate,
                global_step=1,
                walltime=None
            )
            self.writer.add_scalar(
                tag="test/final_balance",
                scalar_value=final_balance,
                global_step=1,
                walltime=None
            )
            self.writer.add_scalar(
                tag="test/required_money",
                scalar_value=required_money,
                global_step=1,
                walltime=None
            )
            self.writer.add_scalar(
                tag="test/reward_sum",
                scalar_value=reward_sum,
                global_step=1,
                walltime=None
            )
            self.writer.add_scalar(
                tag="test/commission_fee",
                scalar_value=np.sum(commission_fee_list),
                global_step=1,
                walltime=None
            )
            
            # 立即刷新 TensorBoard writer，确保数据写入磁盘
            self.writer.flush()
            print("✅ Metrics written to TensorBoard")
            sys.stdout.flush()
            
            # 打印测试结果
            print(f"\n{'='*80}")
            print(f"Test Results")
            print(f"{'='*80}")
            print(f"  Return Rate:      {return_rate:.4f}")
            print(f"  Final Balance:    {final_balance:.2f}")
            print(f"  Required Money:   {required_money:.2f}")
            print(f"  Reward Sum:       {reward_sum:.4f}")
            print(f"  Commission Fee:   {np.sum(commission_fee_list):.4f}")
            print(f"  Total Steps:      {step_count}")
            print(f"{'='*80}\n")
            sys.stdout.flush()
            
            self.writer.close()
            return return_rate
        
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            self.writer.close()
            return None


def main():
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"High-Level Model Standalone Testing Script")
    print(f"{'='*80}")
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Num actions: {args.num_actions}")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    try:
        tester = HighLevelTester(args)
        return_rate = tester.test()
        
        if return_rate is not None:
            print(f"\n✅ Testing completed successfully!")
            print(f"   Results saved to: {tester.output_dir}")
            return 0
        else:
            print(f"\n❌ Testing failed!")
            return 1
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
