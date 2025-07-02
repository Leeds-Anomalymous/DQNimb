import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
# from sklearn.model_selection import train_test_split

class ImbalancedDataset:
    def __init__(self, dataset_name="mnist", rho=0.01, batch_size=64, seed=42):
        """
        初始化数据集处理类
        :param dataset_name: 数据集名称 (e.g., "mnist", "cifar10")
        :param rho: 不平衡因子 (正类样本数 = rho * 负类样本数)
        :param batch_size: DataLoader 批次大小
        :param seed: 随机种子（确保可复现）
        """
        self.dataset_name = dataset_name
        self.rho = rho
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
            self.positive_classes = [2]  # MNIST中少数类的标签（例如：数字2）
            self.negative_classes = [i for i in range(10) if i not in self.positive_classes]  # MNIST中默认将除正类外的所有标签设为负类
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)) #对每个像素进行归一化，0.1307和0.3081分别是MNIST训练集的均值和标准差。这样可以让模型训练更稳定、收敛更快。
            ])
            
            print("正在下载MNIST训练集...")
            train_set = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            
            print("正在下载MNIST测试集...")
            test_set = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            return train_set, test_set
        elif self.dataset_name == "cifar10":
            # CIFAR-10支持
            self.positive_classes = [1]  # 汽车类
            self.negative_classes = [3, 4, 5, 6]  # 指定CIFAR10中的负类标签
            # 数据转换
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), 
                    (0.2470, 0.2435, 0.2616)
                )
            ])

            # 设置CIFAR-10下载源为亚马逊S3镜像
            torchvision.datasets.CIFAR10.url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
            
            print("正在下载CIFAR-10训练集...")
            train_set = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            
            print("正在下载CIFAR-10测试集...")
            test_set = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            return train_set, test_set
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
        # 获取标签数据 - 处理不同数据集的标签格式
        if isinstance(self.train_data.targets, list):
            train_labels = np.array(self.train_data.targets)
        else:
            train_labels = self.train_data.targets.numpy()
            
        if isinstance(self.test_data.targets, list):
            test_labels = np.array(self.test_data.targets)
        else:
            test_labels = self.test_data.targets.numpy()
        
        # 降采样训练集
        selected_data, remapped_labels = self._downsample_data(
            self.train_data.data, train_labels, self.positive_classes, self.negative_classes
        )
        
        # 确保数据是torch张量
        if not isinstance(selected_data, torch.Tensor):
            selected_data = torch.tensor(selected_data)
        
        self.train_data = TensorDataset(selected_data, torch.tensor(remapped_labels))
        
        # 处理测试集（仅重映射标签，不降采样）
        # 只选择正类和负类标签的数据
        valid_indices = np.where(np.isin(test_labels, self.positive_classes) | np.isin(test_labels, self.negative_classes))[0]
        test_data = self.test_data.data[valid_indices]
        test_labels = test_labels[valid_indices]
        
        remapped_test_labels = np.where(
            np.isin(test_labels, self.positive_classes), 0, 1
        )
        
        # 确保测试数据是torch张量
        if not isinstance(test_data, torch.Tensor):
            test_data = torch.tensor(test_data)
            
        self.test_data = TensorDataset(
            test_data, 
            torch.tensor(remapped_test_labels)
        )

    def _downsample_data(self, data, labels, positive_classes, negative_classes):
        """
        专门用于降采样的方法
        :param data: 原始数据
        :param labels: 原始标签
        :param positive_classes: 正类标签值列表
        :param negative_classes: 负类标签值列表
        :return: (降采样后的数据, 重映射后的标签)
        """
        # Step 1: 分离正/负类索引
        positive_idx = np.where(np.isin(labels, positive_classes))[0]
        negative_idx = np.where(np.isin(labels, negative_classes))[0]
        n_negative = len(negative_idx)  # 负类原始样本数 N
        
        # Step 2: 降采样正类（目标样本数 = rho * N）
        positive_num = max(1, int(self.rho * n_negative))  # 至少保留1个样本
        downsampled_positive_idx = np.random.choice(
            positive_idx, size=positive_num, replace=False
        )
        
        # Step 3: 合并降采样后的正类 + 全部负类
        selected_idx = np.concatenate([downsampled_positive_idx, negative_idx])
        np.random.shuffle(selected_idx)  # 打乱顺序
        
        # Step 4: 创建新数据集（标签映射：正类->0, 负类->1）
        selected_data = data[selected_idx]
        selected_labels = labels[selected_idx]
        remapped_labels = np.where(
            np.isin(selected_labels, positive_classes), 0, 1  # 少数类=0, 多数类=1
        )
        
        return selected_data, remapped_labels

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
# Example usage:
if __name__ == "__main__":
    from datasets import ImbalancedDataset

    # 初始化 MNIST 数据集（rho=0.01, 正类=标签2）
    dataset = ImbalancedDataset(dataset_name="cifar10", rho=0.01, batch_size=64)

    # 获取 DataLoader
    train_loader, test_loader = dataset.get_dataloaders()

    # 验证类别分布
    dist = dataset.get_class_distribution()
    print(f"Train distribution: {dist['train']}")  # e.g., [540, 54042] for rho=0.01
    print(f"Test distribution: {dist['test']}")     # e.g., [1032, 8968]