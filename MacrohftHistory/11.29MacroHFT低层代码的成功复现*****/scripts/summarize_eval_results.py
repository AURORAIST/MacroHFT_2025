#!/usr/bin/env python
"""
汇总所有评估结果
读取所有评估生成的 .npy 文件并生成对比表格
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

def summarize_evaluation_results(base_path="./result/low_level/ETHUSDT"):
    """汇总所有评估结果"""
    
    eval_dirs = [d for d in os.listdir(base_path) if d.startswith("eval_")]
    
    if not eval_dirs:
        print("未找到评估结果文件夹")
        return
    
    print(f"\n找到 {len(eval_dirs)} 个评估结果文件夹\n")
    
    results = []
    
    for eval_dir in sorted(eval_dirs):
        eval_path = os.path.join(base_path, eval_dir)
        
        # 提取模型信息
        parts = eval_dir.split("_")
        if len(parts) >= 3:
            model_name = "_".join(parts[1:-1])  # 去掉 "eval" 和 "val/test"
            eval_set = parts[-1]
        else:
            model_name = eval_dir
            eval_set = "unknown"
        
        # 查找 .npy 文件
        npy_files = [f for f in os.listdir(eval_path) if f.endswith('.npy')]
        
        if not npy_files:
            continue
        
        # 读取所有 .npy 文件
        for npy_file in npy_files:
            file_path = os.path.join(eval_path, npy_file)
            
            try:
                data = np.load(file_path)
                
                # 确定文件类型
                if 'return' in npy_file.lower():
                    metric = 'return_rate'
                elif 'reward' in npy_file.lower():
                    metric = 'reward'
                elif 'action' in npy_file.lower():
                    metric = 'action'
                else:
                    metric = 'unknown'
                
                # 提取初始仓位信息（如果有）
                init_pos = None
                if 'init' in npy_file:
                    # 从文件名中提取初始仓位
                    for part in npy_file.split('_'):
                        if part.isdigit():
                            init_pos = int(part)
                            break
                
                # 计算统计信息
                if len(data) > 0:
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    max_val = np.max(data)
                    min_val = np.min(data)
                    
                    results.append({
                        'model': model_name,
                        'eval_set': eval_set,
                        'metric': metric,
                        'init_pos': init_pos,
                        'mean': mean_val,
                        'std': std_val,
                        'max': max_val,
                        'min': min_val,
                        'count': len(data)
                    })
            
            except Exception as e:
                print(f"读取文件出错 {file_path}: {e}")
    
    if not results:
        print("未找到有效的评估数据")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 筛选 return_rate 结果
    df_return = df[df['metric'] == 'return_rate'].copy()
    
    if len(df_return) > 0:
        # 转换为百分比
        for col in ['mean', 'std', 'max', 'min']:
            df_return[col] = df_return[col] * 100
        
        # 按模型和评估集分组
        print("="*100)
        print("评估结果汇总 - 收益率 (%)")
        print("="*100)
        
        # 创建透视表
        pivot_data = []
        for model in df_return['model'].unique():
            for eval_set in ['val', 'test']:
                model_data = df_return[(df_return['model'] == model) & (df_return['eval_set'] == eval_set)]
                
                if len(model_data) > 0:
                    # 如果有多个初始仓位,取平均
                    mean_return = model_data['mean'].mean()
                    std_return = model_data['std'].mean()
                    max_return = model_data['max'].max()
                    min_return = model_data['min'].min()
                    count = model_data['count'].sum()
                    
                    pivot_data.append({
                        '模型': model,
                        '数据集': eval_set,
                        '平均收益率(%)': f"{mean_return:.4f}",
                        '标准差(%)': f"{std_return:.4f}",
                        '最大收益率(%)': f"{max_return:.4f}",
                        '最小收益率(%)': f"{min_return:.4f}",
                        '样本数': int(count)
                    })
        
        summary_df = pd.DataFrame(pivot_data)
        
        if len(summary_df) > 0:
            # 按模型排序
            summary_df = summary_df.sort_values(['模型', '数据集'])
            
            # 打印表格
            print(summary_df.to_string(index=False))
            
            # 保存到CSV
            output_file = os.path.join(base_path, "evaluation_summary.csv")
            summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n结果已保存到: {output_file}")
            
            # 找出最佳模型
            print("\n" + "="*100)
            print("最佳模型")
            print("="*100)
            
            # 验证集最佳
            val_data = summary_df[summary_df['数据集'] == 'val'].copy()
            val_data['平均收益率_数值'] = val_data['平均收益率(%)'].astype(float)
            if len(val_data) > 0:
                best_val = val_data.loc[val_data['平均收益率_数值'].idxmax()]
                print(f"\n验证集最佳: {best_val['模型']}")
                print(f"  平均收益率: {best_val['平均收益率(%)']}%")
                print(f"  标准差: {best_val['标准差(%)']}%")
            
            # 测试集最佳
            test_data = summary_df[summary_df['数据集'] == 'test'].copy()
            test_data['平均收益率_数值'] = test_data['平均收益率(%)'].astype(float)
            if len(test_data) > 0:
                best_test = test_data.loc[test_data['平均收益率_数值'].idxmax()]
                print(f"\n测试集最佳: {best_test['模型']}")
                print(f"  平均收益率: {best_test['平均收益率(%)']}%")
                print(f"  标准差: {best_test['标准差(%)']}%")
            
            print("\n" + "="*100)
    else:
        print("未找到收益率数据")

if __name__ == "__main__":
    summarize_evaluation_results()
