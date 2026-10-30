import torch.distributed.rpc as rpc
import threading 
import time
import argparse
import pathlib
import sys
import queue
ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".") 
from RL.util.logging_config import setup_logger  # 修改日志配置导入
from config import global_workersize,action_dim,n_state_1_dim,n_state_2_dim

from model.net import *

class ParameterServer:
    def __init__(self, global_workersize=1):
        # 初始化日志记录器，添加角色信息
        self.logger = setup_logger("parameter_server", "distributed_dqn.log", role="ParameterServer")
        self.logger.info(f"ParameterServer初始化，global_workersize: {global_workersize}") 
         
        self.model = hyperagent(n_state_1_dim, n_state_2_dim, action_dim, 32).to("cpu")
        
        self.learning_in_progress = False
        self.condition = threading.Condition()
        self.global_workersize = global_workersize

        # 添加全局batchsize相关属性
        self.completed_local_batches = 0
        self.batch_completion_lock = threading.Lock()
        self.eval_index = 0
        self.eval_queue = queue.Queue(100)
        # 注册RPC
        self.logger.info("ParameterServer RPC注册完成")

        
        
    def get_parameters(self):
        """Worker拉取最新模型参数"""
        self.logger.debug("Worker拉取模型参数")
        return [p.detach().clone() for p in self.model.parameters()]

    def update_parameters(self, new_params):
        """Learner推送更新后的参数"""
        self.logger.info("Learner推送更新后的参数")
        for old_p, new_p in zip(self.model.parameters(), new_params):
            old_p.data.copy_(new_p)
        self.logger.debug("参数更新完成")

    def set_learning_status(self, status):
        """设置学习状态"""
        with self.condition:
            self.learning_in_progress = status
            self.logger.info(f"设置学习状态: {status}")             

    def get_learning_status(self):
        """获取学习状态"""
        with self.condition:
            self.logger.debug(f"获取学习状态: {self.learning_in_progress}")
            return self.learning_in_progress
        
    def should_start_learning(self):
        """检查是否所有Worker都已完成本地批次"""
        with self.batch_completion_lock:
            return self.completed_local_batches >= self.global_workersize
        
    def report_local_batch_completion(self):
        """Worker报告本地批次完成"""
        with self.batch_completion_lock:
            self.completed_local_batches += 1
            self.logger.info(f"Worker报告本地批次完成，已完成批次数: {self.completed_local_batches}/{self.global_workersize}")
        
    def reset_batch_completion(self):
        """重置批次完成计数"""
        with self.batch_completion_lock:
            self.completed_local_batches = 0
            self.logger.info("批次完成计数已重置")

    def get_eval_params(self):
        """Eval quque拉取评估参数"""
        self.logger.debug("Eval quque拉取评估参数")
        try:
            eval_params_dict = self.eval_queue.get(True,1)
            if eval_params_dict is not None:
                return eval_params_dict["i"], eval_params_dict["p"]
        except queue.Empty:
            return None,None
        
        return None,None
    
    def push_eval_params(self):
        """Eval quque推送评估参数"""
        self.eval_index += 1
        self.eval_queue.put({"i": self.eval_index, "p": [p.detach().clone() for p in self.model.parameters()]})
        self.logger.debug("Eval quque推送评估参数")
        return self.eval_index


    @staticmethod
    def run(rank, host, world_size):
        # 初始化RPC，确保指向正确的机器和端口
        print("ParameterServer run start")
        rpc.init_rpc(
            "parameter_server",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{host}:29500"
            )
        ) 
         
        print("ParameterServer已启动")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("收到中断信号，ParameterServer停止运行")
        # 保持RPC运行
        rpc.shutdown()
        
# 全局变量用于存储远程对象的RRef
remote_obj_rref = None 
global_lock = threading.Lock()

def get_remote_parameter_server():
    """
    获取远程对象的RRef
    """
    global remote_obj_rref 
    with global_lock:
        if remote_obj_rref is None:
            server = ParameterServer(global_workersize=global_workersize)
            remote_obj_rref = server
        return remote_obj_rref


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.0.109", help="host for distributed training")
    parser.add_argument("--world_size", type=int, default=5, help="world size for distributed training") 
    args = parser.parse_args()
    
    world_size = args.world_size
    host = args.host    
    
    ParameterServer.run(0, host, world_size)