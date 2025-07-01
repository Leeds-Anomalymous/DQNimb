
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from MNIST import MNIST


class MyRL():
    def __init__(self):
    
        self.env = MNIST()
        self.threshold = 1e-4
        self.max_steps = 5000
        self.discount_factor = 0.9
        self.episode_length = 200
        self.episode_needed = 125
        
        #以下是对于数据类的初步规划
        self.Biggroup = []
        self.Smallgroup = []
        self.bili= len(self.Smallgroup) / len(self.Biggroup) if len(self.Biggroup) > 0 else 1


    def Reward(self, state, action, label):
        """奖励函数"""
        iterminal = False
        # 检查动作是否为终止状态
        if state in self.Biggroup:
            if action == label:
                reward = 1.0
            else:
                reward = -1.0
                iterminal = True
        elif state in self.Smallgroup:
            if action == label:
                reward = 0.5
            else:
                reward = -0.5
                iterminal = True
        # 这里可以根据具体任务定义奖励函数
        # 例如，返回一个随机值作为奖励
        return (reward, iterminal)
    
    def DQN(self):
        """深度Q网络实现"""
        # DQN超参数
        epochs = 5000
        losses = []
        mem_size = 1000
        batch_size = 200
        replay = deque(maxlen=mem_size)

        max_moves =50
        h = 0
        sync_freq = 500#设置更新频率
        

        

