
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
    def __init__(self, input_shape, rho=0.01):

        self.discount_factor = 0.1
        self.mem_size = 50000
        self.rho= rho
        self.lambda_value = rho 
        self.t_max = 120000
        self.eta = 0.05
        self.learning_rate = 0.00025
        self.batch_size = 64
                
        # 初始化双网络
        self.q_net = Q_Net_image(input_shape, output_dim=2) #在线网络，实时更新 - 二分类输出
        self.target_net = Q_Net_image(input_shape, output_dim=2) #目标网络，用来软更新 - 二分类输出
        self.target_net.load_state_dict(self.q_net.state_dict())  # 同步参数

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        # 经验回放池
        self.replay_memory = deque(maxlen=self.mem_size)

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 训练计数器
        self.step_count = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (self.t_max * 0.8)

    def compute_reward(self, action, label):
        """
        实现论文中的奖励函数 (Section 3.2)
        Args:
            action: 预测的类别 (0或1)
            label: 真实的类别 (0表示少数类，1表示多数类)
        Returns:
            reward: 奖励值
            terminal: 是否终止当前episode
        """
        terminal = False
        # 少数类样本 (标签0)
        if label == 0:
            if action == label:
                reward = 1.0  # 正确分类少数类
            else:
                reward = -1.0  # 错误分类少数类
                terminal = True  # 终止当前episode
        # 多数类样本 (标签1)
        else:
            if action == label:
                reward = self.lambda_value  # 正确分类多数类
            else:
                reward = -self.lambda_value  # 错误分类多数类
                # 注意: 多数类错误不终止episode
        return reward, terminal

    def replay_experience(self):
        """从经验回放缓冲区采样并训练网络"""
        if len(self.replay_memory) < self.batch_size:
            return
                
        # 随机采样一批经验
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)

        # 将3D张量列表转换为4D批次张量 [batch_size, C, H, W]
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.bool, device=self.device).unsqueeze(1)
            
        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions)
            
        # 计算目标Q值
        with torch.no_grad():
           next_q = self.target_net(next_states).max(1, keepdim=True)[0]
           target_q = rewards + self.discount_factor * next_q * (~terminals)
            
        # 计算损失并更新
        loss = F.mse_loss(current_q, target_q)

        # 清零梯度，反向传播，更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        # 更新目标网络 (软更新)
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.eta * param.data + (1.0 - self.eta) * target_param.data)
            
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    

    def train(self, train_data, train_labels):
        """
        训练DQN分类器 (实现论文Algorithm 2)
        Args:
            train_data: 训练数据张量 (N, C, H, W)
            train_labels: 训练标签张量 (N,)
        """
        num_samples = len(train_data)
        self.step_count = 0
        
        # 训练直到达到最大步数
        while self.step_count < self.t_max:
            # 打乱训练数据 
            indices = torch.randperm(num_samples)
            shuffled_data = train_data[indices]
            shuffled_labels = train_labels[indices]
            
            # 初始化状态 - 不需要添加批次维度，因为数据已经是4D的
            state = shuffled_data[0:1]  # 取第一个样本，保持4D形状 [1, C, H, W]
            
            # 遍历数据集
            for t in range(num_samples - 1):  # 注意: 最后一个样本没有next_state
                # 选择动作 (ε-greedy策略)
                if random.random() < self.epsilon:
                    action = random.randint(0, 1)  # 随机探索
                else:
                    with torch.no_grad():
                        q_values = self.q_net(state.to(self.device))
                    action = q_values.argmax().item()
                
                # 获取奖励和终止标志 
                reward, terminal = self.compute_reward(action, shuffled_labels[t].item())
                
                # 下一个状态 - 同样保持4D形状
                next_state = shuffled_data[t+1:t+2]  # [1, C, H, W]
                
                # 存储经验时去掉批次维度，存储单个样本
                self.replay_memory.append((
                    state.squeeze(0).clone().detach().cpu(),  # 去掉批次维度 [C, H, W]
                    action,
                    reward,
                    next_state.squeeze(0).clone().detach().cpu(),  # 去掉批次维度 [C, H, W]
                    terminal
                ))
                
                # 训练更新 
                self.replay_experience()
                self.step_count += 1
                
                # 检查终止条件 
                if terminal:
                    break
                
                # 更新状态
                state = next_state
                
                # 检查是否达到最大步数
                if self.step_count >= self.t_max:
                    break
            
            # 打印训练进度
            if self.step_count % 1000 == 0:
                print(f"Step: {self.step_count}/{self.t_max}, Epsilon: {self.epsilon:.4f}")
        
        print("Training completed!")
    

    def get_dataloaders(self, dataset_name, rho=0.01, batch_size=64):
        """
        生成训练和测试 DataLoader
        :return: (train_loader, test_loader)
        """
        dataset = ImbalancedDataset(dataset_name=dataset_name, rho=rho, batch_size=batch_size)
        train_loader, test_loader = dataset.get_dataloaders()
        return train_loader, test_loader
    
def main():
    # 创建不平衡数据集
    dataset = ImbalancedDataset(dataset_name="mnist", rho=0.01, batch_size=64)
        
    # 获取整个训练集 (用于DQN序列训练)
    train_data, train_labels = dataset.train_data.tensors
    # 确保数据是正确的形状: (N, 1, 28, 28)
    if len(train_data.shape) == 3:  # (N, 28, 28)
        train_data = train_data.unsqueeze(1)  # 添加通道维度 -> (N, 1, 28, 28)
    train_data = train_data.float()
    
    # 初始化DQN分类器
    input_shape = (1, 28, 28)  # 输入形状: 通道, 高度, 宽度
    classifier = MyRL(input_shape, rho=0.01)
    
    # 开始训练
    classifier.train(train_data, train_labels)

if __name__ == "__main__":
    main()
        
                

            

                 
        

        

