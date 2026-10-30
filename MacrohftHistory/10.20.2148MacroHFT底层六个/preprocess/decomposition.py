import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 定义一个函数用于平滑数据，使用巴特沃斯低通滤波器
def smooth_data(data):
    N, Wn = 1, 0.05  # N为滤波器阶数，Wn为归一化的截止频率
    b, a = butter(N, Wn, btype='low')  # 设计巴特沃斯低通滤波器
    return filtfilt(b, a, data)  # 应用滤波器

# 定义一个函数用于获取平滑数据的斜率
def get_slope(smoothed_data):
    X = np.arange(len(smoothed_data)).reshape(-1, 1)  # 创建自变量数组
    model = LinearRegression().fit(X, smoothed_data)  # 拟合线性回归模型
    return model.coef_[0]  # 返回斜率

# 定义一个函数用于获取窗口数据的斜率，内部调用smooth_data和get_slope
def get_slope_window(window):
    N, Wn = 1, 0.05  # 同smooth_data函数中的参数
    b, a = butter(N, Wn, btype='low')  # 设计巴特沃斯低通滤波器
    y = filtfilt(b, a, window.values)  # 对窗口数据进行平滑处理
    X = np.arange(len(y)).reshape(-1, 1)  # 创建自变量数组
    model = LinearRegression().fit(X, y)  # 拟合线性回归模型
    return model.coef_[0]  # 返回斜率

# 定义一个函数用于将数据集分块并存储为feather格式
def chunk(df_train, df_val, df_test):
    chunk_size = 4320  # 定义每个数据块的大小
    # 对训练集进行分块
    for i in range(int(len(df_train) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        df_chunk = df_train[start:end].reset_index(drop=True)  # 提取并重置索引
        df_chunk.to_feather('./data/ETHUSDT/train/df_{}.feather'.format(i))  # 保存为feather文件
    # 对验证集进行分块
    for i in range(int(len(df_val) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        df_chunk = df_val[start:end].reset_index(drop=True)  # 提取并重置索引
        df_chunk.to_feather('./data/ETHUSDT/val/df_{}.feather'.format(i))  # 保存为feather文件
    # 对测试集进行分块
    for i in range(int(len(df_test) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        df_chunk = df_test[start:end].reset_index(drop=True)  # 提取并重置索引
        df_chunk.to_feather('./data/ETHUSDT/test/df_{}.feather'.format(i))  # 保存为feather文件

# 定义一个函数用于标记数据块的斜率，并根据斜率分组
def label_slope(df_train, df_val, df_test):
    chunk_size = 4320  # 定义每个数据块的大小
    slopes_train = []  # 初始化存储训练集斜率的列表
    # 计算训练集每个数据块的斜率
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        chunk = df_train['close'][start:end].values  # 提取收盘价数据
        smoothed_chunk = smooth_data(chunk)  # 平滑处理数据
        slope = get_slope(smoothed_chunk)  # 计算斜率
        slopes_train.append(slope)  # 将斜率添加到列表中

    slopes_val = []  # 初始化存储验证集斜率的列表
    # 计算验证集每个数据块的斜率
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        chunk = df_val['close'][start:end].values  # 提取收盘价数据
        smoothed_chunk = smooth_data(chunk)  # 平滑处理数据
        slope = get_slope(smoothed_chunk)  # 计算斜率
        slopes_val.append(slope)  # 将斜率添加到列表中

    slopes_test = []  # 初始化存储测试集斜率的列表
    # 计算测试集每个数据块的斜率
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        chunk = df_test['close'][start:end].values  # 提取收盘价数据
        smoothed_chunk = smooth_data(chunk)  # 平滑处理数据
        slope = get_slope(smoothed_chunk)  # 计算斜率
        slopes_test.append(slope)  # 将斜率添加到列表中

    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]  # 定义分位数列表
    # 使用qcut方法对训练集斜率进行分箱，并返回分箱边界
    slope_labels_train, bins = pd.qcut(slopes_train, q=quantiles, retbins=True, labels=False)
    train_indices = [[] for _ in range(5)]  # 初始化存储训练集索引的列表
    # 将训练集每个数据块的斜率分组到对应的索引列表中
    for index, label in enumerate(slope_labels_train):
        train_indices[label].append(index)
    # 使用pickle模块保存训练集索引分组结果
    with open('./data/ETHUSDT/train/slope_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    bins[0] = -100  # 手动调整分箱边界
    bins[-1] = 100  # 手动调整分箱边界
    # 使用cut方法对验证集斜率进行分箱
    slope_labels_val = pd.cut(slopes_val, bins=bins, labels=False, include_lowest=True)
    # 对验证集斜率的分组结果进行调整，使得标签在1到3之间
    slope_labels_val = [1 if element == 0 else element for element in slope_labels_val]
    slope_labels_val = [3 if element == 4 else element for element in slope_labels_val]
    val_indices = [[] for _ in range(5)]  # 初始化存储验证集索引的列表
    # 将验证集每个数据块的斜率分组到对应的索引列表中
    for index, label in enumerate(slope_labels_val):
        val_indices[label].append(index)
    # 使用pickle模块保存验证集索引分组结果
    with open('./data/ETHUSDT/val/slope_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)

    # 使用cut方法对测试集斜率进行分箱
    slope_labels_test = pd.cut(slopes_test, bins=bins, labels=False, include_lowest=True)
    # 对测试集斜率的分组结果进行调整，使得标签在1到3之间
    slope_labels_test = [1 if element == 0 else element for element in slope_labels_test]
    slope_labels_test = [3 if element == 4 else element for element in slope_labels_test]
    test_indices = [[] for _ in range(5)]  # 初始化存储测试集索引的列表
    # 将测试集每个数据块的斜率分组到对应的索引列表中
    for index, label in enumerate(slope_labels_test):
        test_indices[label].append(index)
    # 使用pickle模块保存测试集索引分组结果
    with open('./data/ETHUSDT/test/slope_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)

# 定义一个函数用于标记数据块的波动性，并根据波动性分组
def label_volatility(df_train, df_val, df_test):
    chunk_size = 4320  # 定义每个数据块的大小
    volatilities_train = []  # 初始化存储训练集波动性的列表
    # 计算训练集每个数据块的波动性
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        chunk = df_train[start:end]  # 提取数据块
        chunk['return'] = chunk['close'].pct_change().fillna(0)  # 计算收益率并填充缺失值
        volatility = chunk['return'].std()  # 计算波动性
        volatilities_train.append(volatility)  # 将波动性添加到列表中

    volatilities_val = []  # 初始化存储验证集波动性的列表
    # 计算验证集每个数据块的波动性
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        chunk = df_val[start:end]  # 提取数据块
        chunk['return'] = chunk['close'].pct_change().fillna(0)  # 计算收益率并填充缺失值
        volatility = chunk['return'].std()  # 计算波动性
        volatilities_val.append(volatility)  # 将波动性添加到列表中

    volatilities_test = []  # 初始化存储测试集波动性的列表
    # 计算测试集每个数据块的波动性
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size  # 计算每个数据块的起始位置
        end = (i + 1) * chunk_size  # 计算每个数据块的结束位置
        chunk = df_test[start:end]  # 提取数据块
        chunk['return'] = chunk['close'].pct_change().fillna(0)  # 计算收益率并填充缺失值
        volatility = chunk['return'].std()  # 计算波动性
        volatilities_test.append(volatility)  # 将波动性添加到列表中

    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]  # 定义分位数列表
    # 使用qcut方法对训练集波动性进行分箱，并返回分箱边界
    vol_labels_train, bins = pd.qcut(volatilities_train, q=quantiles, retbins=True, labels=False)
    train_indices = [[] for _ in range(5)]  # 初始化存储训练集索引的列表
    # 将训练集每个数据块的波动性分组到对应的索引列表中
    for index, label in enumerate(vol_labels_train):
        train_indices[label].append(index)
    # 使用pickle模块保存训练集索引分组结果
    with open('./data/ETHUSDT/train/vol_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    bins[0] = 0  # 手动调整分箱边界
    bins[-1] = 1  # 手动调整分箱边界
    # 使用cut方法对验证集波动性进行分箱
    vol_labels_val = pd.cut(volatilities_val, bins=bins, labels=False, include_lowest=True)
    # 对验证集波动性的分组结果进行调整，使得标签在1到3之间
    vol_labels_val = [1 if element == 0 else element for element in vol_labels_val]
    vol_labels_val = [3 if element == 4 else element for element in vol_labels_val]
    val_indices = [[] for _ in range(5)]  # 初始化存储验证集索引的列表
    # 将验证集每个数据块的波动性分组到对应的索引列表中
    for index, label in enumerate(vol_labels_val):
        val_indices[label].append(index)
    # 使用pickle模块保存验证集索引分组结果
    with open('./data/ETHUSDT/val/vol_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)

    # 使用cut方法对测试集波动性进行分箱
    vol_labels_test = pd.cut(volatilities_test, bins=bins, labels=False, include_lowest=True)
    # 对测试集波动性的分组结果进行调整，使得标签在1到3之间
    vol_labels_test = [1 if element == 0 else element for element in vol_labels_test]
    vol_labels_test = [3 if element == 4 else element for element in vol_labels_test]
    test_indices = [[] for _ in range(5)]  # 初始化存储测试集索引的列表
    # 将测试集每个数据块的波动性分组到对应的索引列表中
    for index, label in enumerate(vol_labels_test):
        test_indices[label].append(index)
    # 使用pickle模块保存测试集索引分组结果
    with open('./data/ETHUSDT/test/vol_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)

# 定义一个函数用于为整个数据集添加斜率和波动性标签
def label_whole(df):
    window_size_list = [360]  # 定义窗口大小列表
    for i in range(len(window_size_list)):
        window_size = window_size_list[i]  # 获取窗口大小
        # 计算每个数据块的斜率，并添加到数据集中
        df['slope_{}'.format(window_size)] = df['close'].rolling(window=window_size).apply(get_slope_window)
        df['return'] = df['close'].pct_change().fillna(0)  # 计算收益率并填充缺失值
        # 计算每个数据块的波动性，并添加到数据集中
        df['vol_{}'.format(window_size)] = df['return'].rolling(window=window_size).std()
    return df  # 返回添加了斜率和波动性标签的数据集

if __name__ == "__main__":
    print("开始")  # 打印开始信息
    # 读取训练集、验证集和测试集的feather文件
    df_train = pd.read_feather('./data/ETHUSDT/df_train.feather')
    df_val = pd.read_feather('./data/ETHUSDT/df_val.feather')
    df_test = pd.read_feather('./data/ETHUSDT/df_test.feather')
    print("完成数据读取")  # 打印完成信息

    # 创建存储结果的文件夹
    os.makedirs('./data/ETHUSDT/train', exist_ok=True)
    os.makedirs('./data/ETHUSDT/val', exist_ok=True)
    os.makedirs('./data/ETHUSDT/test', exist_ok=True)
    os.makedirs('./data/ETHUSDT/whole', exist_ok=True)
    print("完成文件夹创建")  # 打印完成信息

    # 对训练集、验证集和测试集进行分块
    chunk(df_train, df_val, df_test)
    # 标记训练集、验证集和测试集的斜率，并保存索引分组结果
    label_slope(df_train, df_val, df_test)
    # 标记训练集、验证集和测试集的波动性，并保存索引分组结果
    label_volatility(df_train, df_val, df_test)

    # 对训练集、验证集和测试集添加斜率和波动性标签，并处理缺失值
    df_train = label_whole(df_train).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_val = label_whole(df_val).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_test = label_whole(df_test).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    print("完成标记")  # 打印完成信息

    # 将处理后的数据集保存为feather文件
    df_train.to_feather('./data/ETHUSDT/whole/train.feather')
    df_val.to_feather('./data/ETHUSDT/whole/val.feather')
    df_test.to_feather('./data/ETHUSDT/whole/test.feather')
    print("处理完成")  # 打印结束信息
