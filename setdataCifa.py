import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from skimage import color
from skimage.feature import local_binary_pattern
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

class Cifar10Imbalanced:
    def __init__(self, root='./data', rho=0.4, batch_size=64, 
                 lbp_radius=1, lbp_points=8, seed=42):
        """
        CIFAR-10不平衡数据集处理（汽车类 vs 动物类）
        :param root: 数据存储路径
        :param rho: 不平衡因子（正类样本数 = rho * 负类样本数）
        :param batch_size: DataLoader批次大小
        :param lbp_radius: LBP特征提取半径
        :param lbp_points: LBP邻域采样点数
        :param seed: 随机种子
        """
        self.root = root
        self.rho = rho
        self.batch_size = batch_size
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 定义类别映射
        self.positive_class = 1  # 汽车类
        self.negative_classes = [3, 4, 5, 6]  # 猫、鹿、狗、青蛙
        
        # 加载并处理数据
        self._load_data()
        self._extract_features()
        self._create_imbalanced_dataset()

    def _load_data(self):
        """加载CIFAR-10数据集并筛选相关类别"""
        print("正在下载CIFAR-10数据集...")
        # 完整数据集用于筛选
        full_train = CIFAR10(root=self.root, train=True, download=True)
        full_test = CIFAR10(root=self.root, train=False, download=True)
        
        # 筛选训练集
        train_labels = np.array(full_train.targets)
        train_pos_idx = np.where(train_labels == self.positive_class)[0]
        train_neg_idx = np.where(np.isin(train_labels, self.negative_classes))[0]
        
        # 筛选测试集
        test_labels = np.array(full_test.targets)
        test_pos_idx = np.where(test_labels == self.positive_class)[0]
        test_neg_idx = np.where(np.isin(test_labels, self.negative_classes))[0]
        
        # 提取图像数据
        self.train_pos_images = full_train.data[train_pos_idx]
        self.train_neg_images = full_train.data[train_neg_idx]
        self.test_pos_images = full_test.data[test_pos_idx]
        self.test_neg_images = full_test.data[test_neg_idx]
        
        print(f"训练集正类样本数: {len(self.train_pos_images)}")
        print(f"训练集负类样本数: {len(self.train_neg_images)}")
        print(f"测试集正类样本数: {len(self.test_pos_images)}")
        print(f"测试集负类样本数: {len(self.test_neg_images)}")

    def _extract_features(self):
        """为所有样本提取LBP特征"""
        print("提取训练集正类LBP特征...")
        train_pos_features = self._extract_lbp_histograms(self.train_pos_images)
        print("提取训练集负类LBP特征...")
        train_neg_features = self._extract_lbp_histograms(self.train_neg_images)
        print("提取测试集正类LBP特征...")
        test_pos_features = self._extract_lbp_histograms(self.test_pos_images)
        print("提取测试集负类LBP特征...")
        test_neg_features = self._extract_lbp_histograms(self.test_neg_images)
        
        # 保存特征
        self.train_pos_features = train_pos_features
        self.train_neg_features = train_neg_features
        self.test_pos_features = test_pos_features
        self.test_neg_features = test_neg_features

    def _extract_lbp_histograms(self, images):
        """对图像集合提取LBP直方图特征"""
        features = []
        for img in images:
            # 转换为灰度图
            gray_img = color.rgb2gray(img)
            
            # 计算LBP特征
            lbp = local_binary_pattern(
                gray_img, 
                P=self.lbp_points, 
                R=self.lbp_radius, 
                method='uniform'
            )
            
            # 计算直方图（bins = lbp_points + 2）
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, self.lbp_points + 3),
                range=(0, self.lbp_points + 2),
                density=True  # 归一化直方图
            )
            features.append(hist)
        
        return np.array(features)

    def _create_imbalanced_dataset(self):
        """创建不平衡数据集（根据rho降采样正类）"""
        # 计算降采样后的正类样本数
        n_neg = len(self.train_neg_features)
        n_pos = max(1, int(self.rho * n_neg))
        
        # 随机选择正类样本
        pos_indices = np.random.choice(
            len(self.train_pos_features), 
            size=n_pos, 
            replace=False
        )
        
        # 合并特征和标签
        train_features = np.concatenate([
            self.train_pos_features[pos_indices],
            self.train_neg_features
        ])
        train_labels = np.concatenate([
            np.zeros(n_pos),  # 正类标签为0
            np.ones(n_neg)    # 负类标签为1
        ])
        
        # 创建TensorDataset
        self.train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)
        )
        
        # 测试集（不降采样）
        test_features = np.concatenate([
            self.test_pos_features,
            self.test_neg_features
        ])
        test_labels = np.concatenate([
            np.zeros(len(self.test_pos_features)),
            np.ones(len(self.test_neg_features))
        ])
        
        self.test_dataset = TensorDataset(
            torch.tensor(test_features, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long)
        )
        
        print(f"训练集最终样本数: {len(train_labels)} (正类: {n_pos}, 负类: {n_neg})")
        print(f"测试集样本数: {len(test_labels)}")

    def get_dataloaders(self):
        """
        生成训练和测试DataLoader
        :return: (train_loader, test_loader)
        """
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        return train_loader, test_loader

    def get_feature_dimension(self):
        """返回特征维度"""
        return self.lbp_points + 2

    def get_class_distribution(self):
        """返回处理后的类别分布"""
        train_labels = self.train_dataset.tensors[1].numpy()
        test_labels = self.test_dataset.tensors[1].numpy()
        return {
            "train": np.bincount(train_labels.astype(int)),
            "test": np.bincount(test_labels.astype(int))
        }


from setdataCifa import Cifar10Imbalanced

# 创建高度不平衡数据集（正类只有负类的1%）
dataset = Cifar10Imbalanced(rho=0.04, batch_size=64)

# 获取数据加载器
train_loader, test_loader = dataset.get_dataloaders()

# 查看类别分布
dist = dataset.get_class_distribution()
print(f"训练集分布: 正类={dist['train'][0]}, 负类={dist['train'][1]}")
print(f"测试集分布: 正类={dist['test'][0]}, 负类={dist['test'][1]}")