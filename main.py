TEST_ONLY = False  # 设置为 True 时只进行评估，不进行训练

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
        self.batch_size = 128
                
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
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (self.t_max*0.6)

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

    def replay_experience(self, update_target=True):
        """从经验回放缓冲区采样并训练网络"""                
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
            
        # 更新目标网络 (软更新)，只在update_target为True时更新
        # 更新参数 φ := (1-η)φ + ηθ
        if update_target:
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.eta * param.data + (1.0 - self.eta) * target_param.data)
            
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            

    def train(self, dataset):
        """
        按照论文Algorithm 2训练DQN分类器
        Args:
            dataset: 数据集对象
        """
        # 获取完整数据集
        train_data, train_labels, _, _ = dataset.get_full_dataset()
        
        self.step_count = 0
        episode = 0
        
        # 创建总体训练进度条
        total_pbar = tqdm(total=self.t_max, desc="Training Progress", unit="step")
        
        # 外层循环: for episode k = 1 to K do (直到达到最大步数)
        while self.step_count < self.t_max:
            episode += 1
            
            # 打乱训练数据顺序 (Shuffle the training data D)
            indices = torch.randperm(len(train_data))
            shuffled_data = train_data[indices]
            shuffled_labels = train_labels[indices]
            
            # 初始化状态 s_1 = x_1
            current_state = shuffled_data[0:1]
            if len(current_state.shape) == 3:
                current_state = current_state.unsqueeze(1)  # 添加通道维度
            current_state = current_state.float().to(self.device)
            
            # 修正通道顺序
            if current_state.shape[1] != 3 and current_state.shape[-1] == 3:
                current_state = current_state.permute(0, 3, 1, 2)  # NHWC -> NCHW
            
            # 进度条显示
            episode_pbar = tqdm(
                total=len(shuffled_data)-1,  # 最后一个样本没有next_state
                desc=f"Episode {episode}", 
                leave=False, 
                unit="sample"
            )
            
            # 内层循环: for t = 1 to T do (遍历所有样本)
            for t in range(len(shuffled_data) - 1):
                # 检查是否已达到最大步数
                if self.step_count >= self.t_max:
                    break
                
                # 获取当前标签
                current_label = shuffled_labels[t].item()
                
                # 根据ε-greedy策略选择动作 (Choose an action based on ε-greedy policy)
                if random.random() < self.epsilon:
                    action = random.randint(0, 1)  # 随机探索
                else:
                    with torch.no_grad():
                        q_values = self.q_net(current_state)
                    action = q_values.argmax().item()
                
                # 计算奖励和终止标志 (r_t, terminal_t = STEP(a_t, l_t))
                reward, terminal = self.compute_reward(action, current_label)
                
                # 获取下一状态 (Set s_{t+1} = x_{t+1})
                next_state = shuffled_data[t+1:t+2]
                if len(next_state.shape) == 3:
                    next_state = next_state.unsqueeze(1)
                next_state = next_state.float().to(self.device)
                
                # 修正通道顺序
                if next_state.shape[1] != 3 and next_state.shape[-1] == 3:
                    next_state = next_state.permute(0, 3, 1, 2)
                
                # 存储经验到记忆库 (Store (s_t, a_t, r_t, s_{t+1}, terminal_t) to M)
                self.replay_memory.append((
                    current_state.squeeze(0).cpu().clone().detach(),
                    action,
                    reward,
                    next_state.squeeze(0).cpu().clone().detach(),
                    terminal
                ))
                
                # 从记忆库中采样并学习(仅当记忆库足够大时)
                if len(self.replay_memory) >= self.batch_size:
                    # 根据terminal状态决定是否更新目标网络
                    self.replay_experience(update_target=not terminal)

                    self.step_count += 1
                    total_pbar.update(1)
                
                # 更新进度条
                episode_pbar.update(1)
                episode_pbar.set_postfix({
                    'Step': self.step_count,
                    'Epsilon': f'{self.epsilon:.4f}',
                    'Reward': f'{reward:.4f}',
                    'Terminal': terminal
                })
                
                # 如果是terminal状态，则终止当前episode
                if terminal:
                    break
                    
                # 设置当前状态为下一状态，继续循环
                current_state = next_state
            
            episode_pbar.close()
            
            # 显示episode信息
            total_pbar.set_postfix({
                'Episode': episode,
                'Epsilon': f'{self.epsilon:.4f}',
                'Memory': len(self.replay_memory)
            })
        
        total_pbar.close()
        print("训练完成!")
    
    
def main():
    # 创建不平衡数据集
    dataset = ImbalancedDataset(dataset_name="mnist", rho=0.0005, batch_size=64)
        
    # 直接获取训练和测试的dataloader
    train_loader, test_loader = dataset.get_dataloaders()
    
    # 初始化DQN分类器
    input_shape = (1, 28, 28)  # 输入形状: 通道, 高度, 宽度
    
    # 创建checkpoints目录（如果不存在）
    os.makedirs('checkpoints', exist_ok=True)
    model_path = os.path.join('checkpoints', 'dqn_classifier.pth')
    
    if TEST_ONLY:
        print("测试模式: 仅加载模型并评估")
        # 创建模型但不训练
        q_net = Q_Net_image(input_shape, output_dim=2)
        
        # 加载预训练模型
        if os.path.exists(model_path):
            q_net.load_state_dict(torch.load(model_path))
            print(f"成功加载模型: {model_path}")
            
            # 评估模型
            evaluate_model(q_net, test_loader, save_dir='checkpoints')
        else:
            print(f"错误: 未找到预训练模型 {model_path}")
            print("请先将 TEST_ONLY 设置为 False 进行训练，或确保模型文件存在")
    else:
        print("训练模式: 将进行模型训练和评估")
        classifier = MyRL(input_shape, rho=0.0005)
        
        # 开始训练，直接使用数据集对象而不是dataloader
        classifier.train(dataset)
        
        # 生成带编号的模型文件名
        base_name = 'dqn_classifier'
        counter = 1
        while True:
            model_filename = f'{base_name}_{counter}.pth'
            numbered_model_path = os.path.join('checkpoints', model_filename)
            if not os.path.exists(numbered_model_path):
                break
            counter += 1
        
        # 保存模型
        torch.save(classifier.q_net.state_dict(), numbered_model_path)
        print(f"模型已保存到 {numbered_model_path}")
        
        # 评估模型
        evaluate_model(classifier.q_net, test_loader, save_dir='checkpoints')


if __name__ == "__main__":
    main()










