
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
        

        

