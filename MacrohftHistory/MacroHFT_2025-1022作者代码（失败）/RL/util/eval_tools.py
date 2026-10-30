from collections import deque
import pandas as pd
import numpy as np
import logging as log
import statsmodels.api as sm

def _generate_trade_profit_loss_records(trade_records):
    """
    生成交易盈亏记录
    
    根据买卖记录，使用最近的卖出记录抵消买入记录来生成交易盈亏记录。
    允许1笔买入用多笔卖出抵消。
    
    返回:
        DataFrame: 包含交易盈亏记录的数据框，列包括：
            - datetime: 交易时间
            - buy_id: 买入交易ID
            - sell_id: 卖出交易ID
            - buy_price: 买入价格
            - sell_price: 卖出价格
            - amount: 交易金额
            - profit_loss: 盈亏金额
            - quantity: 交易数量
    """
    # 分离买入和卖出记录
    buy_records = [record for record in trade_records if record['type'] == 'buy']
    sell_records = [record for record in trade_records if record['type'] == 'sell']
    
    # 按时间排序
    buy_records.sort(key=lambda x: x['datetime'])
    sell_records.sort(key=lambda x: x['datetime'])
    
    profit_loss_records = []
    
    # 使用队列处理买入记录
    
    buy_queue = deque(buy_records)
    
    # 处理每笔卖出记录
    for sell_record in sell_records:
        remaining_sell_quantity = sell_record['quantity']
        
        while remaining_sell_quantity > 0 and buy_queue:
            buy_record = buy_queue[0]
            
            # 计算可抵消的数量
            if buy_record['quantity'] <= remaining_sell_quantity:
                # 整笔买入记录被完全抵消
                matched_quantity = buy_record['quantity']
                buy_queue.popleft()  # 移除已完全抵消的买入记录
            else:
                # 部分抵消买入记录
                matched_quantity = remaining_sell_quantity
                # 更新买入记录的剩余数量
                buy_record['quantity'] -= matched_quantity
                buy_record['amount'] = buy_record['quantity'] * buy_record['price'] * (1 + buy_record['commission_fee'] / (buy_record['quantity'] * buy_record['price']))
            
            # 计算盈亏
            buy_price = buy_record['price']
            sell_price = sell_record['price']
            profit_loss = (sell_price - buy_price) * matched_quantity - buy_record['commission_fee'] * (matched_quantity / buy_record['quantity']) - sell_record['commission_fee'] * (matched_quantity / sell_record['quantity'])
            
            # 计算交易金额
            amount = sell_price * matched_quantity
            
            # 添加到盈亏记录
            profit_loss_records.append({
                'datetime': sell_record['datetime'],
                'buy_id': buy_record['id'],
                'sell_id': sell_record['id'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'amount': amount,
                'profit_loss': profit_loss,
                'quantity': matched_quantity
            })
            
            # 更新剩余卖出数量
            remaining_sell_quantity -= matched_quantity
    
    # 转换为DataFrame并返回
    return pd.DataFrame(profit_loss_records)

def _profit_loss_statistics(profit_loss_records):
    """
    生成日/月维度交易统计
    
    根据交易盈亏记录，生成日维度和月维度的交易统计信息。
    
    返回:
        tuple: (daily_stats, monthly_stats)
            - daily_stats: 日维度交易统计DataFrame
            - monthly_stats: 月维度交易统计DataFrame
            每个DataFrame包含以下列：
            - date/month: 日期/月份
            - trade_count: 交易次数
            - total_amount: 交易金额
            - total_quantity: 交易数量
            - profit_count: 盈利次数
            - profit_amount: 盈利金额
            - loss_amount: 亏损金额
    """
    # 获取交易盈亏记录
 
    
    if profit_loss_records.empty:
        # 如果没有交易记录，返回空的DataFrame
        daily_columns = ['date', 'trade_count', 'total_amount', 'total_quantity', 'profit_count', 'profit_amount', 'loss_amount']
        monthly_columns = ['month', 'trade_count', 'total_amount', 'total_quantity', 'profit_count', 'profit_amount', 'loss_amount']
        return pd.DataFrame(columns=daily_columns), pd.DataFrame(columns=monthly_columns)
    
    # 转换datetime列为datetime类型
    profit_loss_records['datetime'] = pd.to_datetime(profit_loss_records['datetime'])
    
    # 提取日期和月份
    profit_loss_records['date'] = profit_loss_records['datetime'].dt.date
    profit_loss_records['month'] = profit_loss_records['datetime'].dt.to_period('M')
    
    # 计算每笔交易的盈亏状态
    profit_loss_records['is_profit'] = profit_loss_records['profit_loss'] > 0
    
    # 日维度统计
    daily_stats = profit_loss_records.groupby('date').agg(
        trade_count=('profit_loss', 'count'),
        total_amount=('amount', 'sum'),
        total_quantity=('quantity', 'sum'),
        profit_count=('is_profit', 'sum'),
        profit_amount=('profit_loss', lambda x: x[x > 0].sum()),
        loss_amount=('profit_loss', lambda x: abs(x[x < 0].sum()))
    ).reset_index()
    
    # 确保盈利金额和亏损金额在没有盈利或亏损时为0
    daily_stats['profit_amount'] = daily_stats['profit_amount'].fillna(0)
    daily_stats['loss_amount'] = daily_stats['loss_amount'].fillna(0)
    
    # 月维度统计
    monthly_stats = profit_loss_records.groupby('month').agg(
        trade_count=('profit_loss', 'count'),
        total_amount=('amount', 'sum'),
        total_quantity=('quantity', 'sum'),
        profit_count=('is_profit', 'sum'),
        profit_amount=('profit_loss', lambda x: x[x > 0].sum()),
        loss_amount=('profit_loss', lambda x: abs(x[x < 0].sum()))
    ).reset_index()
    
    # 确保盈利金额和亏损金额在没有盈利或亏损时为0
    monthly_stats['profit_amount'] = monthly_stats['profit_amount'].fillna(0)
    monthly_stats['loss_amount'] = monthly_stats['loss_amount'].fillna(0)
    
    # 将月份转换为字符串格式
    monthly_stats['month'] = monthly_stats['month'].astype(str)
    
    return daily_stats, monthly_stats

def _calculate_trading_days(money_history):
    """
    计算交易天数
    
    基于资金历史记录，计算交易天数。
    
    参数:
        money_history: 资金历史记录列表，每个元素为 [timestamp, current_money]
    
    返回:
        int: 交易天数
    """
    if not money_history:
        return 0
    
    # 将时间戳转换为日期，以便计算不同的天数
    dates = set()
    for record in money_history:
        # 假设时间戳是毫秒级的时间戳
        timestamp = record[0]
        # 转换为日期（不考虑时间部分）
        date = pd.to_datetime(timestamp, unit='ms').date()
        dates.add(date)
    
    # 返回不同的日期数量，即交易天数
    return len(dates)

def _calculate_trading_daily_returns(money_history):
    """
    计算交易每日收益率
    
    基于资金历史记录，计算每日收益率。
    
    参数:
        money_history: 资金历史记录列表，每个元素为 [timestamp, current_money]
    
    返回:
        list: 每日收益率列表
    """
    if not money_history or len(money_history) < 2:
        return []
    
    # 按日期分组，计算每日的最终资金
    daily_money = {}
    for record in money_history:
        timestamp, money = record
        # 转换为日期（不考虑时间部分）
        date = pd.to_datetime(timestamp, unit='ms').date()
        
        # 如果该日期还没有记录，或者当前记录的资金更大，则更新
        if date not in daily_money or money > daily_money[date]:
            daily_money[date] = money
    
    # 按日期排序
    sorted_dates = sorted(daily_money.keys())
    
    # 计算每日收益率
    daily_returns = []
    for i in range(1, len(sorted_dates)):
        prev_date = sorted_dates[i-1]
        curr_date = sorted_dates[i]
        prev_money = daily_money[prev_date]
        curr_money = daily_money[curr_date]
        
        # 计算收益率：(当前资金 - 前一天资金) / 前一天资金
        if prev_money != 0:
            daily_return = (curr_money - prev_money) / prev_money
            daily_returns.append(daily_return)
    
    return daily_returns

def _calculate_trading_monthly_returns(money_history):
    """
    计算交易每月收益率
    
    基于资金历史记录，计算每月收益率。
    
    参数:
        money_history: 资金历史记录列表，每个元素为 [timestamp, current_money]
    
    返回:
        list: 每月收益率列表
    """
    if not money_history or len(money_history) < 2:
        return []
    
    # 按月份分组，计算每月的第一天和最后一天资金
    monthly_first_money = {}  # 每月第一天的资金
    monthly_last_money = {}   # 每月最后一天的资金
    
    for record in money_history:
        timestamp, money = record
        # 转换为月份（年-月格式）
        month = pd.to_datetime(timestamp, unit='ms').to_period('M')
        
        # 如果该月份还没有记录，则初始化
        if month not in monthly_first_money:
            monthly_first_money[month] = money
            monthly_last_money[month] = money
        else:
            # 更新每月最后一天的资金
            monthly_last_money[month] = money
    
    # 按月份排序
    sorted_months = sorted(monthly_first_money.keys())
    
    # 计算每月收益率
    monthly_returns = []
    for month in sorted_months:
        first_money = monthly_first_money[month]
        last_money = monthly_last_money[month]
        
        # 计算收益率：(当月最后一天资金 - 当月第一天资金) / 当月第一天资金
        if first_money != 0:
            monthly_return = (last_money - first_money) / first_money
            monthly_returns.append(monthly_return)
    
    return monthly_returns


def calculate_trading_metrics(df,trade_records,money_history,no_risk_return):
    """
    计算交易指标
    
    基于交易记录计算以下指标：
    - 总交易金额
    - 年化波动率 = 每日收益率标准差
    - 下行标准差 = 低于目标收益率的波动性
    - 交易胜率 = 盈利次数/总交易次数
    - 盈亏比 = 平均每次盈利金额/平均每次亏损金额
    - 总交易次数
    - 交易频率 = 总交易次数/交易天数
    
    参数:   
            df: 市场数据
            trade_records: 交易记录列表
            money_history: 每日资金历史记录列表，每个元素为 [timestamp, current_money]
            no_risk_return: 无风险返回率
    返回:
        dict: 包含计算指标的字典
            - total_amount: 总交易金额
            - annualized_volatility: 年化波动率
            - downside_deviation: 下行标准差
            - win_rate: 交易胜率
            - profit_loss_ratio: 盈亏比
            - total_trades: 总交易次数
            - trade_frequency: 交易频率
    """
    # 获取交易盈亏记录
    profit_loss_records = _generate_trade_profit_loss_records(trade_records)
    daily_stats, monthly_stats = _profit_loss_statistics(profit_loss_records)
    trading_days = _calculate_trading_days(money_history)
    daily_returns_history = _calculate_trading_daily_returns(money_history)
    monthly_returns_history = _calculate_trading_monthly_returns(money_history)
    monthly_market_baseline_returns = _calculate_monthly_market_baseline_return(df)
    monthly_no_risk_returns = _calculate_monthly_no_risk_return(df, no_risk_return)
    if profit_loss_records.empty:
        # 如果没有交易记录，返回默认值
        return {
            'total_amount': 0,
            'annualized_volatility': 0,
            'downside_deviation': 0,
            'win_rate': 0,
            'profit_loss_ratio': 0,
            'total_trades': 0,
            'trade_frequency': 0
        }
        
    log.info(profit_loss_records.head(10))
    log.info(profit_loss_records.tail(10))
    
    # 转换datetime列为datetime类型
    profit_loss_records['datetime'] = pd.to_datetime(profit_loss_records['datetime'])
    
    # 提取日期
    profit_loss_records['date'] = profit_loss_records['datetime'].dt.date
    
    # 计算总交易金额
    total_amount = profit_loss_records['amount'].sum()
    
      
    # 计算年化波动率 = 每日收益率标准差
    daily_returns = np.array(daily_returns_history)
    daily_std = daily_returns.std()
    annualized_volatility = daily_std * np.sqrt(356/trading_days) if not np.isnan(daily_std) else 0
    
    # 计算下行标准差
    downside_deviation = _calculate_downside_deviation(daily_returns)
    
    # 计算交易胜率 = 盈利次数 / 总交易次数
    profit_loss_records['is_profit'] = profit_loss_records['profit_loss'] > 0
    total_trades = len(profit_loss_records)
    profit_trades = profit_loss_records['is_profit'].sum()
    win_rate = profit_trades / total_trades if total_trades > 0 else 0
    
    # 计算盈亏比 = 平均每次盈利金额 / 平均每次亏损金额
    profit_amounts = profit_loss_records[profit_loss_records['profit_loss'] > 0]['profit_loss']
    loss_amounts = profit_loss_records[profit_loss_records['profit_loss'] < 0]['profit_loss'].abs()
    
    avg_profit = profit_amounts.mean() if len(profit_amounts) > 0 else 0
    avg_loss = loss_amounts.mean() if len(loss_amounts) > 0 else 0
    
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0    
    
    # 计算交易频率 = 总交易次数 / 交易天数
    trade_frequency = total_trades / trading_days if trading_days > 0 else 0
    
    # 计算alpha和beta
    alpha, beta  = _calculate_alpha_beta(monthly_returns_history, monthly_market_baseline_returns, monthly_no_risk_returns)

    
    return {
        'total_amount': total_amount,
        'annualized_volatility': annualized_volatility,
        'downside_deviation': downside_deviation,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'total_trades': total_trades,
        'trade_frequency': trade_frequency,
        'alpha': alpha,
        'beta': beta
    }

def _calculate_downside_deviation(returns, target_return=0.02):
    """
    计算下行标准差
    
    下行标准差是一种风险度量指标，只考虑低于目标收益率的波动性。
    
    参数:
        returns (pd.Series or np.array): 收益率序列
        target_return (float): 目标收益率，默认为0
        
    返回:
        float: 下行标准差
    """
    # 确保输入是numpy数组
    returns = np.array(returns)
    
    # 计算低于目标收益率的部分
    downside_returns = returns[returns < target_return] - target_return
    
    # 如果没有低于目标收益率的值，返回0
    if len(downside_returns) == 0:
        return 0
    
    # 计算下行标准差
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    return downside_deviation


def _calculate_monthly_market_baseline_return(df):
    """
    计算市场基准收益率：以第一天价格买入，不卖出，最后一天卖出，计算每个月的收益率
    
    Parameters:
        df (pandas.DataFrame): 数据集，包含至少包含'timestamp'和'close'列
   
        
    Returns:
        dict: 包含每月收益率和总体收益率的字典
    """
    
    
    # 确保数据按时间排序
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    else:
        # 如果没有时间戳列，假设数据已经是按时间排序的
        pass
        
    # 获取第一天和最后一天的收盘价
    first_close = df.iloc[0]['close']
    last_close = df.iloc[-1]['close']
    
    # 计算总体收益率
    total_return = (last_close - first_close) / first_close
    
    # 计算每月收益率
    # 首先将时间戳转换为日期时间格式（如果还不是）
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        # 如果没有时间戳列，创建一个假设的日期索引
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
    # 按月份分组
    df['year_month'] = df['date'].dt.to_period('M')
    
    # 计算每月的第一天和最后一天的价格
    monthly_first = df.groupby('year_month').first()['close']
    monthly_last = df.groupby('year_month').last()['close']
    
    # 计算每月收益率
    monthly_returns = (monthly_last - monthly_first) / monthly_first
    
    return monthly_returns


def _calculate_monthly_no_risk_return(df, no_risk_return):
    """
    根据数据集的月份生成每个月无风险收益
    
    基于给定的年化无风险收益率，计算每个月的无风险收益。
    
    参数:
        df: 包含时间序列数据的DataFrame
        no_risk_return: 年化无风险收益率，默认为2%
        
    返回:
        pd.Series: 包含每个月无风险收益的Series，索引为月份
    """
    # 确保数据按时间排序
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    else:
        # 如果没有时间戳列，假设数据已经是按时间排序的
        pass
        
    # 计算月度无风险收益率
    # 年化收益率转换为月度收益率：(1 + 年化收益率)^(1/12) - 1
    monthly_norisk_return = (1 + no_risk_return / 100) ** (1/12) - 1
    
    # 首先将时间戳转换为日期时间格式（如果还不是）
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        # 如果没有时间戳列，创建一个假设的日期索引
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
    # 按月份分组
    df['year_month'] = df['date'].dt.to_period('M')
    
    # 获取所有唯一的月份
    unique_months = df['year_month'].unique()
    
    # 为每个月创建相同的无风险收益率
    monthly_returns = pd.Series([monthly_norisk_return] * len(unique_months), index=unique_months)
    
    return monthly_returns


def _calculate_alpha_beta(monthly_returns_history, monthly_market_baseline_returns, monthly_no_risk_returns):
    """
    计算策略的alpha和beta
    
    基于策略月度收益率、市场基准月度收益率和无风险月度收益率，
    使用线性回归方法计算策略的alpha和beta指标。
    
    参数:
        monthly_returns_history: 策略月度收益率列表
        monthly_market_baseline_returns: 市场基准月度收益率Series
        monthly_no_risk_returns: 无风险月度收益率Series
    
    返回:
        tuple: (alpha, beta)
            - alpha: 策略alpha值
            - beta: 策略beta值
    """
    # 确保有足够的数据进行计算
    if len(monthly_returns_history) < 2 or len(monthly_market_baseline_returns) < 2:
        return 0, 0
    
    # 将策略收益率转换为numpy数组
    strategy_returns = np.array(monthly_returns_history)
    
    # 获取市场基准收益率和无风险收益率
    # 确保市场基准和无风险收益率与策略收益率长度一致
    market_returns = np.array(monthly_market_baseline_returns.values[-len(strategy_returns):])
    risk_free_returns = np.array(monthly_no_risk_returns.values[-len(strategy_returns):])
    
    # 计算超额收益率（策略收益率 - 无风险收益率）
    strategy_excess_returns = strategy_returns - risk_free_returns
    
    # 计算市场超额收益率（市场收益率 - 无风险收益率）
    market_excess_returns = market_returns - risk_free_returns
    
    # 创建DataFrame以便于处理
    df = pd.DataFrame({
        'strategy_return': strategy_returns,
        'market_return': market_returns,
        'risk_free': risk_free_returns,
        'strategy_excess': strategy_excess_returns,
        'market_excess': market_excess_returns
    })
    
    # 进行线性回归
    # 添加常数项（对应Alpha）
    X = sm.add_constant(df['market_excess'])
    y = df['strategy_excess']
    
    try:
        model = sm.OLS(y, X).fit()
        
        # 提取Alpha和Beta
        alpha = model.params['const']
        beta = model.params['market_excess']
        
        return alpha, beta
    except:
        # 如果回归失败，回退到原来的计算方法
        # 计算beta
        # beta = 协方差(策略超额收益率, 市场超额收益率) / 方差(市场超额收益率)
        if np.var(market_excess_returns) != 0:
            covariance = np.cov(strategy_excess_returns, market_excess_returns)[0, 1]
            beta = covariance / np.var(market_excess_returns)
        else:
            beta = 0
        
        # 计算alpha
        # alpha = 平均策略超额收益率 - beta * 平均市场超额收益率
        alpha = np.mean(strategy_excess_returns) - beta * np.mean(market_excess_returns)
        
        return alpha, beta