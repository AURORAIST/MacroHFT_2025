import logging
import logging.handlers
import os
import sys
from datetime import datetime
import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".") 

# 创建logs目录（如果不存在）
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志格式，包含进程ID、线程ID和角色信息
class RoleFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        
    def format(self, record):
        # 如果没有设置角色，则使用logger名称作为角色
        if not hasattr(record, 'role'):
            record.role = record.name
        return super().format(record)

# 使用自定义格式化器，添加角色信息
formatter = RoleFormatter(
    '%(asctime)s - PID:%(process)d - TID:%(thread)d - Role:%(role)s - %(levelname)s - %(message)s'
)

def setup_logger(name, log_file, level=logging.INFO, role=None):
    """设置并返回一个配置好的logger"""
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 设置角色属性
    logger.role = role if role else name
    
    # 避免重复添加handler
    if not logger.handlers:
        # 文件handler，按大小轮转
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, log_file), 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# 添加一个辅助函数，用于在日志记录时添加角色信息
def log_with_role(logger, level, msg, role=None):
    """使用指定角色记录日志"""
    if role is None:
        role = getattr(logger, 'role', logger.name)
    
    # 创建一个临时的LogRecord来添加角色信息
    extra = {'role': role}
    logger.log(level, msg, extra=extra)