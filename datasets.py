import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split

class ImbalancedDataset:
    def __init__(self, dataset_name="mnist", rho=0.01, positive_class=2, batch_size=64, seed=42):
        """
        初始化数据集处理类
        :param dataset_name: 数据集名称 (e.g., "mnist")
        :param rho: 不平衡因子 (正类样本数 = rho * 负类样本数)
        :param positive_class: 少数类（正类）的原始标签
        :param batch_size: DataLoader 批次大小
        :param seed: 随机种子（确保可复现）
        """
        self.dataset_name = dataset_name
        self.rho = rho
        self.positive_class = positive_class
        self.batch_size = batch_size
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 加载并预处理数据
        self.train_data, self.test_data = self.load_raw_data()
        self._preprocess_data()

    def load_raw_data(self):
        """加载原始数据集（需扩展时在此添加新数据集）"""
        if self.dataset_name == "mnist":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            return train_set, test_set
        elif self.dataset_name == "cifar10":
            # 示例：未来扩展 CIFAR-10
            # transform = ...
            # train_set = torchvision.datasets.CIFAR10(...)
            raise NotImplementedError("CIFAR-10 support coming soon!")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _preprocess_data(self):
        """
        核心预处理：降采样正类 + 重映射标签（0/1）
        遵循论文（Section 4.3）：
          - 正类（少数类）标签 -> 0
          - 负类（多数类）标签 -> 1
          - 正类样本数降至 rho * N（N=负类原始样本数）
        """
        # Step 1: 分离正/负类索引
        train_indices = np.arange(len(self.train_data))
        train_labels = self.train_data.targets.numpy()
        
        positive_idx = np.where(train_labels == self.positive_class)[0]
        negative_idx = np.where(train_labels != self.positive_class)[0]
        n_negative = len(negative_idx)  # 负类原始样本数 N
        
        # Step 2: 降采样正类（目标样本数 = rho * N）
        target_positive = max(1, int(self.rho * n_negative))  # 至少保留1个样本
        downsampled_positive_idx = np.random.choice(
            positive_idx, size=target_positive, replace=False
        )
        
        # Step 3: 合并降采样后的正类 + 全部负类
        selected_idx = np.concatenate([downsampled_positive_idx, negative_idx])
        np.random.shuffle(selected_idx)  # 打乱顺序
        
        # Step 4: 创建新数据集（标签映射：正类->0, 负类->1）
        selected_data = self.train_data.data[selected_idx]
        selected_labels = train_labels[selected_idx]
        remapped_labels = np.where(
            selected_labels == self.positive_class, 0, 1  # 少数类=0, 多数类=1
        )
        
        self.train_data = TensorDataset(selected_data, torch.tensor(remapped_labels))
        
        # Step 5: 处理测试集（仅重映射标签，不降采样）
        test_labels = self.test_data.targets.numpy()
        remapped_test_labels = np.where(
            test_labels == self.positive_class, 0, 1
        )
        self.test_data = TensorDataset(
            self.test_data.data, 
            torch.tensor(remapped_test_labels)
        )

    def get_dataloaders(self):
        """
        生成训练和测试 DataLoader
        :return: (train_loader, test_loader)
        """
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, test_loader

    # 可选：添加其他辅助方法
    def get_class_distribution(self):
        """返回处理后的类别分布（用于验证）"""
        train_labels = self.train_data.tensors[1].numpy()
        test_labels = self.test_data.tensors[1].numpy()
        return {
            "train": np.bincount(train_labels),
            "test": np.bincount(test_labels)
        }