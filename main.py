
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
    
    def DQN(self,size):
        """深度Q网络实现"""
        # DQN超参数
        epochs = 5000
        losses = []
        mem_size = 1000
        batch_size = 200
        gamma = 0.99    # 折扣因子
        epsilon = 1.0   # 初始探索率
        epsilon_min = 0.01
        epsilon_decay = 0.995
        batch_size = 64
        tau = 0.005 
        replay = deque(maxlen=mem_size)


        #初始化 Q网和target网
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化双网络
        q_net = Q_Net_image(size) #在线网络，实时更新
        target_net = Q_Net_image(size) #目标网络，用来软更新
        target_net.load_state_dict(self.q_net.state_dict())  # 同步参数


        optimizer = optim.Adam(q_net.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        def epsilon_greedy_action(state_num, epsilon):
            """ε-贪婪策略选择动作"""
            if random.random() < epsilon:
                return random.randint(0, len(self.env.action_space) - 1)
            else:
                state_tensor = torch.FloatTensor(state_num).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                return q_values.argmax().item()

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
            
            # 计算当前Q值
            current_q = q_net(states).gather(1, actions)
            
            # 计算目标Q值
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                target_q = rewards + gamma * next_q * (~dones)
            
            # 计算损失并更新
            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新目标网络 (软更新)
            for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
            # 衰减探索率
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        

        for i in range(epochs):
            shuffled_indices = np.random.permutation(len(Data))
            states, labels = data[shuffled_indices, :-1], data[shuffled_indices, -1]
            
            s = states[0]  # 初始化状态 
            
            for t in range(len(states) - 1):
            # 当前状态转换为tensor
                state_tensor = torch.FloatTensor(states[t]).unsqueeze(0).to(device)
                
                # ε-贪婪选择动作
                action = epsilon_greedy_action(state_tensor, epsilon)
                
                # 获取奖励和终止标志
                reward, terminal = self.Reward(states[t], action, labels[t])
                
                # 下一个状态
                next_state_tensor = torch.FloatTensor(states[t + 1]).unsqueeze(0).to(device)
                
                # 存储经验到回放缓冲区
                replay.append((state_tensor, action, reward, next_state_tensor, terminal))
                
                
                # 经验回放训练
                replay_experience()
                
                # 如果终止则跳出
                if terminal:
                    break
            
            # 打印训练进度
        
                

            

                 
        

        

