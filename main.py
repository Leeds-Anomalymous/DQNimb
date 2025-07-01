
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

        self.max_steps = 1000
        self.discount_factor = 0.9
        self.mem_size = 5000
        self.rho= 0.01
        self.t_max = 120000

    
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
        epsilon = 1.0   # 初始探索率
        epsilon_min = 0.01
        batch_size = 64
        eta = 0.005 
        replay = deque(maxlen=self.mem_size)


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
                target_q = rewards + self.discount_factor * next_q * (~dones)
            
            # 计算损失并更新
            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新目标网络 (软更新)
            for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                target_param.data.copy_(eta * param.data + (1.0 - eta) * target_param.data)
            
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
    def get_dataloaders(self, dataset_name, rho=0.01, batch_size=64):
        """
        生成训练和测试 DataLoader
        :return: (train_loader, test_loader)
        """
        dataset = ImbalancedDataset(dataset_name=dataset_name, rho=rho, batch_size=batch_size)
        train_loader, test_loader = dataset.get_dataloaders()
        return train_loader, test_loader
        
                

            

                 
        

        

