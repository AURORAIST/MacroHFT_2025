#有2个维度， 
# 第一个维度 代表多头合约持仓，0代表空仓，long
# 第二个维度 代表空头合约持仓，0代表空仓，short
# [1,1] 可以作为保护头寸，这个是基本上是没有利润的，甚至又隔夜持仓成本

long_actions = [
    [0,0],
    [1,0],
 
]

short_actions = [
    [0,0],
    [0,1],
    
]

short_and_long_actions = [   
    [0,0],
    [0,1],
    [1,0], 
    
]

def get_actions(action_mode="both", size=1):
    """
    获取动作列表和动作数量信息
    
    参数:
        action_mode (str): 动作模式，可选值: "long", "short", "both"
            - "long": 多头动作模式
            - "short": 空头动作模式  
            - "both": 多空双向动作模式
        size (int): 动作数量控制参数
            - size=1: 包含前2个动作
            - size=2: 包含前3个动作
            - size=3: 包含前4个动作
            - size=4: 包含前5个动作
            - size=5: 包含前6个动作
            - size=6: 包含前7个动作
            - size=7: 包含所有动作
            - 其他值: 包含所有可用动作
    
    返回:
        tuple: (actions, n_actions, action_type)
            - actions: 动作列表，每个动作是包含两个元素的列表 [long_position, short_position]
            - n_actions: 动作数量
            - action_type: 动作类型描述字符串
    """
    if action_mode == "long":
        actions = long_actions
        action_type = "多头动作模式"
    elif action_mode == "short":
        actions = short_actions
        action_type = "空头动作模式"
    elif action_mode == "both":  # "both" 或其他值默认使用多空双向模式
        actions = short_and_long_actions
        action_type = "多空双向动作模式"
    else:
        raise Exception ("we do not support other action mode yet")
    # 根据size参数控制动作数量
    if size == 1:
        actions = actions[:2]  # 包含前2个动作
    elif size == 2:
        actions = actions[:3]  # 包含前3个动作
    elif size == 3:
        actions = actions[:4]  # 包含前4个动作
    elif size == 4:
        actions = actions[:5]  # 包含前5个动作
    elif size == 5:
        actions = actions[:6]  # 包含前6个动作
    elif size == 6:
        actions = actions[:7]  # 包含前7个动作
    elif size == 7:
        actions = actions  # 包含所有动作
    # 其他size值保持所有动作不变
    
    n_actions = len(actions)
    
    return actions, n_actions, action_type

# # 更新示例用法以测试size参数
if __name__ == "__main__":
    # 测试不同size参数
    print("=== 测试size参数功能 ===")
    for size_val in [1]:
        print(f"\nsize={size_val} 时的动作列表:")
        for mode in [ "both"]:
            actions, n_actions, action_type = get_actions(mode, size=size_val)
            print(f"动作模式: {mode}")
            print(f"动作类型: {action_type}")
            print(f"动作数量: {n_actions}")
            print(f"动作列表: {actions}")
            print("-" * 30)