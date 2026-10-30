#!/bin/bash

# 杀死所有包含"macrohft"的Python进程
# 此脚本用于终止所有运行中与MacroHFT相关的Python进程

echo "正在查找包含'macrohft'的Python进程..."

# 查找所有包含"macrohft"的Python进程ID
pids=$(ps aux | grep -i "python" | grep -i "RL/agent" | grep -v grep | awk '{print $2}')

# 检查是否找到相关进程
if [ -z "$pids" ]; then
    echo "未找到包含'macrohft'的Python进程"
    exit 0
fi

# 显示找到的进程
echo "找到以下包含'macrohft'的Python进程:"
ps aux | grep -i "python" | grep -i "RL/agent" | grep -v grep

# 询问用户是否要杀死这些进程
read -p "是否要杀死这些进程? (y/n): " confirm
if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    # 杀死进程
    echo "正在杀死进程..."
    echo "$pids" | xargs kill -9
    
    # 验证进程是否已被杀死
    sleep 1
    remaining=$(ps aux | grep -i "python" | grep -i "RL/agent" | grep -v grep | wc -l)
    
    if [ "$remaining" -eq 0 ]; then
        echo "所有包含'macrohft'的Python进程已成功终止"
    else
        echo "警告: 仍有 $remaining 个进程未终止"
        ps aux | grep -i "python" | grep -i "macrohft" | grep -v grep
    fi
else
    echo "操作已取消"
fi