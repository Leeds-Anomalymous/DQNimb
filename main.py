
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datasets import ImbalancedDataset
from Model import Q_Net_image



class MyRL():
    def __init__(self):
    
        self.env = ImbalancedDataset()
        self.threshold = 1e-4
        self.max_steps = 5000
        self.discount_factor = 0.9
        self.episode_length = 200
        self.episode_needed = 125
        
        #以下是对于数据类的初步规划
        self.Biggroup = []
        self.Smallgroup = []
        self.bili= len(self.Smallgroup) / len(self.Biggroup) if len(self.Biggroup) > 0 else 1

    def epsilon_greedy_action(state_num, epsilon):
        """ε-贪婪策略选择动作"""
        if random.random() < epsilon:
            return random.randint(0, len(self.env.action_space) - 1)
        else:
            state_tensor = state_to_tensor(state_num).unsqueeze(0)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            return q_values.argmax().item()
    
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
                reward = self.bili
            else:
                reward = self.bili



        return (reward, iterminal)
    
    def DQN(self,Data):
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
        j=0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_network = DQNNetwork(input_size, hidden_size, output_size).to(device)
        target_network = DQNNetwork(input_size, hidden_size, output_size).to(device)

        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        def replay_experience():
            """从经验回放缓冲区采样并训练网络"""
            if len(replay) < batch_size:
                return
                
            # 随机采样一批经验
            batch = random.sample(replay, batch_size)
            states = torch.cat([s1 for (s1, a, r, s2, d) in batch])
            actions = torch.tensor([a for (s1, a, r, s2, d) in batch])
            rewards = torch.tensor([r for (s1, a, r, s2, d) in batch])
            next_states = torch.cat([s2 for (s1, a, r, s2, d) in batch])
            dones = torch.tensor([d for (s1, a, r, s2, d) in batch])
        

        for i in range(epochs):
            state = self.Data[0][0]
            for t in range(len(Data)):

            

                 
        

        

