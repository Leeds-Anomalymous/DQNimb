import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import os
from datasets import ImbalancedDataset
from Model import Q_Net_image
from evaluate import evaluate_model  # 导入评估模块


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
        
        # 将网络移动到设备
        self.q_net.to(self.device)
        self.target_net.to(self.device)

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
            return False  # 返回False表示没有执行经验回放
                
        # 随机采样一批经验
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)

        # 将数据移动到正确的设备
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
            
        return True  # 返回True表示成功执行了经验回放

    def train(self, train_loader):
        """
        使用dataloader训练DQN分类器 (实现论文Algorithm 2)
        Args:
            train_loader: 训练数据的DataLoader
        """
        self.step_count = 0
        
        # 创建总体训练进度条
        total_pbar = tqdm(total=self.t_max, desc="Training Progress", unit="step")
        
        # 训练直到达到最大步数
        epoch = 0
        while self.step_count < self.t_max:
            epoch += 1
            
            # 创建当前epoch的进度条
            epoch_pbar = tqdm(train_loader, 
                            desc=f"Epoch {epoch}", 
                            leave=False, 
                            unit="batch",
                            total=len(train_loader))
            
            # 遍历数据集
            for data, labels in epoch_pbar:
                # 确保数据是正确的形状: (N, C, H, W)
                if len(data.shape) == 3:  # (N, 28, 28)
                    data = data.unsqueeze(1)  # 添加通道维度 -> (N, 1, 28, 28)
                data = data.float().to(self.device)
                # 修正通道顺序
                if data.shape[1] != 3 and data.shape[-1] == 3:
                    data = data.permute(0, 3, 1, 2)  # NHWC -> NCHW
                labels = labels.to(self.device)
                
                # 处理批次中的每个样本
                for i in range(len(data) - 1):  # 最后一个样本没有next_state
                    # 当前状态
                    state = data[i:i+1]  # 保持4D形状 [1, C, H, W]
                    
                    # 选择动作 (ε-greedy策略)
                    if random.random() < self.epsilon:
                        action = random.randint(0, 1)  # 随机探索
                    else:
                        with torch.no_grad():
                            q_values = self.q_net(state)
                        action = q_values.argmax().item()
                    
                    # 获取奖励和终止标志 
                    reward, terminal = self.compute_reward(action, labels[i].item())
                    
                    # 下一个状态
                    next_state = data[i+1:i+2]  # [1, C, H, W]
                    
                    # 存储经验 - 移动到CPU以节省GPU内存
                    self.replay_memory.append((
                        state.squeeze(0).cpu().clone().detach(),
                        action,
                        reward,
                        next_state.squeeze(0).cpu().clone().detach(),
                        terminal
                    ))
                    
                    # 训练更新 - 只有当成功执行经验回放时才增加步数计数
                    if self.replay_experience():
                        self.step_count += 1
                        total_pbar.update(1)
                    
                    # 更新进度条信息
                    epoch_pbar.set_postfix({
                        'Step': self.step_count,
                        'Epsilon': f'{self.epsilon:.4f}',
                        'Reward': f'{reward:.2f}',
                        'Action': action,
                        'Terminal': terminal
                    })
                    total_pbar.set_postfix({
                        'Epoch': epoch,
                        'Epsilon': f'{self.epsilon:.4f}',
                        'Memory': len(self.replay_memory)
                    })
                    
                    # 检查终止条件 
                    if terminal:
                        break
                    
                    # 检查是否达到最大步数
                    if self.step_count >= self.t_max:
                        break
                if terminal:
                    break  
                # 检查是否达到最大步数
                if self.step_count >= self.t_max:
                    break
            
            epoch_pbar.close()
            
            # 检查是否达到最大步数
            if self.step_count >= self.t_max:
                break
        
        total_pbar.close()
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
    dataset = ImbalancedDataset(dataset_name="cifar10", rho=0.01, batch_size=64)
        
    # 直接获取训练和测试的dataloader
    train_loader, test_loader = dataset.get_dataloaders()
    
    # 初始化DQN分类器
    input_shape = (3, 32, 32)  # 输入形状: 通道, 高度, 宽度
    classifier = MyRL(input_shape, rho=0.01)
    
    # 开始训练，使用dataloader
    classifier.train(train_loader)
    
    # 创建checkpoints目录（如果不存在）
    os.makedirs('checkpoints', exist_ok=True)
    
    # 保存模型
    model_path = os.path.join('checkpoints', 'dqn_classifier.pth')
    torch.save(classifier.q_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 评估模型
    evaluate_model(classifier.q_net, test_loader, save_dir='checkpoints')


if __name__ == "__main__":
    main()










