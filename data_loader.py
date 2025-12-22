import torch
import os
import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import defaultdict
from tqdm import tqdm
from .add_noise import add_mixed_noise
class TBMbearingDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data = torch.tensor(self.Data[index], dtype=torch.float32).to(device)
        label = torch.tensor([self.Label[index]], dtype=torch.long).squeeze().to(device)
        return data, label


class DataPrepare(object):
    def __init__(self,train_data_path,data_path,args,):
        self.train_data_path = train_data_path
        self.data_path = data_path
        self.label_dic = args.label_dic
        self.type_standard = args.type_standard
        self.len_window = args.len_window
        self.stride = args.stride
        self.test_size = args.test_size
        self.batch_size = args.batch_size 
        self.use_val = args.use_val
        self.val_size = args.val_size

        #LTsettings:
        self.use_longtail = args.use_longtail
        self.num_classes = args.num_classes
        self.imb = args.imb
        self.n_max = args.n_max

        #add_noise
        self.add_noise = args.add_noise
        self.snr = args.snr   #信噪比
        self.impulse_prob = args.impulse_prob  #添加冲击噪声频率
        self.impulse_amp = args.impulse_amp   #添加冲击噪声强度

        self.args = args

        # 提高程序健壮性，若使用指定各类样本数下采样的方式，则不使用长尾采样方式
        if self.args.down_sampling == True:
            self.use_longtail = False

        os.makedirs(self.train_data_path, exist_ok=True)

    def split_data(self, data, labels):
        """分割数据为训练集和测试集。"""
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=self.test_size, random_state=1342, stratify=labels) #保持类别间比例

        # 将 NumPy 数组转换为 PyTorch 张量   #RuntimeError: CUDA error: device-side assert triggered
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return X_train, X_test, y_train, y_test

    def time_window(self, window_data, window_label, df_data, label):
        '''
        size window: [num_windows, len_window * num_channel]
        size label : [num_windows, 1]
        '''
        i = 0

        # 计算总迭代次数（用于进度条的total参数）
        total_iterations = (len(df_data) - self.len_window) // self.stride + 1
        with tqdm(total=total_iterations, desc="Processing Sliding Windows") as pbar:
            while i + self.len_window <= len(df_data):
                window_data.append(df_data.iloc[i:i + self.len_window].values)
                window_label.append(int(label))
                i = i + self.stride  # overlap
                pbar.update(1)
        return window_data, window_label

    def scale_data(self, data, scale_style='standard'):
        # 假设data已经是numpy数组，如果不是，则取消下面的注释
        data = np.array(data, dtype=np.float32)
        # 三维数据转为二维，加速标准化
        num_samples, num_steps, num_features = data.shape
        data = data.reshape(-1, num_features)  # 减少数据复制，直接操作原数组

        # 对每个特征进行全局标准化
        # scaler.fit(data)
        if scale_style == 'standard':
            # scaler.transform(data, copy=False)  # 使用in-place操作减少内存使用
            scaler = StandardScaler()
            scaler.fit(data)
            data[:] = scaler.transform(data)  # 使用 NumPy 来实现原地操作减少内存使用
        elif scale_style == 'min-max':
            # data[:] = scaler.transform(data) #使用 NumPy 来实现原地操作减少内存使用(TypeError: MinMaxScaler.transform() got an unexpected keyword argument 'copy' )
            scaler = MinMaxScaler()
            scaler.fit(data)
            data[:] = scaler.transform(data)

        elif scale_style == 'double':  # 双重归一化/串联归一化
            # 先应用标准化
            scaler_standard = StandardScaler()
            scaler_standard.fit(data)
            data[:] = scaler_standard.transform(data)

            # 然后应用min-max归一化
            scaler_min_max = MinMaxScaler(feature_range=(0, 1))
            scaler_min_max.fit(data)
            data[:] = scaler_min_max.transform(data)

        # 还原成原始的三维形状
        data = data.reshape(num_samples, num_steps, num_features)

        print('Data has been scaled!')
        return data

    def get_time_window_from_files(self):
        window_data, window_label = [], []
        files = [file for file in os.listdir(self.data_path) if file != '.ipynb_checkpoints']
        for file in files:
            print('folder_name:',file)
            label = self.label_dic[file]
            print('label:', label)
            if file == '数据说明.txt':
                continue
            file_path1 = os.path.join(self.data_path, file + '/加速度/')
            files2 = os.listdir(file_path1) #['1R-10KN', '1R-20KN', '1R-30KN', '2R-10KN', '2R-20KN', '2R-30KN', '3R-10KN', '3R-20KN', '3R-30KN']
            print('now condition:', files2)
            for file2 in files2:
                file_path2 = os.path.join(file_path1, file2)
                print(file_path2)
                files3 = os.listdir(file_path2) #['X.xlsx', 'Y.xlsx', 'Z.xlsx']
                data = pd.DataFrame()
                for file3 in files3:
                    # 读取第一个sheet，获取数据和列名。
                    df_1= pd.read_excel(os.path.join(file_path2, file3),sheet_name=0,index_col=0, header=[0, 1],
                                                 engine='openpyxl')
                    df_1.reset_index(drop=True, inplace=True)
                    df_1.columns = [file3[0]]
                    # 读取第二个sheet，header = None，并指定列名与第一个sheet相同。
                    df_2 = pd.read_excel(os.path.join(file_path2, file3), sheet_name=1, index_col=0, header=None,
                                             engine='openpyxl')
                    if not df_2.empty:
                        df_2.columns = df_1.columns
                        df_2.reset_index(drop=True, inplace=True)
                        df_2.columns = [file3[0]]
                        # 拼接两个数据框，注意处理索引和列名的匹配。
                        df_3 = pd.concat([df_1, df_2], axis=0, ignore_index=True)
                    else:
                        df_3 = df_1

                    data[file3[0]] = df_3.iloc[:, 0]

                window_data, window_label = self.time_window(window_data, window_label, data, label)
        return window_data, window_label

    def manual_undersample(self, window_data, window_label):
        healthy_class = 0
        fault_classes = np.unique(window_label[window_label != healthy_class])

        # 打印原始数据中各类别的样本数目
        unique, counts = np.unique(window_label, return_counts=True)
        print("Original label distribution:", dict(zip(unique, counts)))

        # 提取健康样本
        X_healthy = window_data[window_label == healthy_class]
        y_healthy = window_label[window_label == healthy_class]

        # 初始化 num_samples_for_faults 以防止引用前未定义
        num_samples_for_faults = []

        if self.args.num_samples[0] == -1:
            X_healthy_resampled = X_healthy
            y_healthy_resampled = y_healthy
        else:
            num_healthy = int(self.args.num_samples[0])
            if len(X_healthy) > num_healthy:
                X_healthy_resampled, y_healthy_resampled = resample(
                    X_healthy, y_healthy, replace=False, n_samples=num_healthy, random_state=42)
            else:
                X_healthy_resampled, y_healthy_resampled = X_healthy, y_healthy
        num_samples_for_faults = self.args.num_samples[1:]  # 使用预设的故障样本数目

        X_resampled = X_healthy_resampled
        y_resampled = y_healthy_resampled

        # 对每个故障类别进行下采样
        for fault_class, num_samples_for_fault in zip(fault_classes, num_samples_for_faults):
            X_fault = window_data[window_label == fault_class]
            y_fault = window_label[window_label == fault_class]

            if num_samples_for_fault == -1:
                X_fault_resampled, y_fault_resampled = X_fault, y_fault
            else:
                if len(X_fault) > num_samples_for_fault:
                    X_fault_resampled, y_fault_resampled = resample(
                        X_fault, y_fault, replace=False, n_samples=num_samples_for_fault, random_state=42)
                else:
                    X_fault_resampled, y_fault_resampled = X_fault, y_fault

            X_resampled = np.vstack([X_resampled, X_fault_resampled])
            y_resampled = np.hstack([y_resampled, y_fault_resampled])

        print(f"Resampled label distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def load_h5(self, file_path):
        with h5py.File(file_path, 'r') as h5f:
            window_data = h5f['data'][:]  # 加载 window_data 数据集
            window_label = h5f['labels'][:]  # 加载 window_label 数据集
        return window_data, window_label

    def save_to_h5(self, data, labels, file_name):
        """将数据和标签保存到HDF5文件中。"""
        with h5py.File(file_name, 'w') as h5f:
            h5f.create_dataset('data', data=data)
            h5f.create_dataset('labels', data=labels)
        del data, labels  # 清理数据

    def _data_loader(self, X_train, y_train, X_test_balanced, y_test_balanced, X_test_imbalanced, y_test_imbalanced):
        test_balanced_dataset = TBMbearingDataset(X_test_balanced, y_test_balanced)
        if self.use_val:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size,
                                                              random_state=None, stratify=y_train)
            print('train time window shape:', X_train.shape)
            print('train label shape:', y_train.shape)
            print('valid time window shape:', X_val.shape)
            print('valid label shape:', y_val.shape)
            val_dataset = TBMbearingDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        elif not self.use_val:
            val_loader = None
            print('train time window shape:', X_train.shape)
            print('train label shape:', y_train.shape)

        train_dataset = TBMbearingDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)
        print('balanced_test time window shape:', X_test_balanced.shape)
        print('balanced_test label shape:', y_test_balanced.shape)
        test_balanced_loader = DataLoader(test_balanced_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        if X_test_imbalanced is not None:
            test_imbalanced_dataset = TBMbearingDataset(X_test_imbalanced, y_test_imbalanced)
            test_imbalanced_loader = DataLoader(test_imbalanced_dataset, batch_size=self.batch_size, shuffle=False,
                                                drop_last=False)
            print('imbalanced_test time window shape:', X_test_imbalanced.shape)
            print('imbalanced_test label shape:', y_test_imbalanced.shape)
        else:
            test_imbalanced_loader = None

        return train_loader, val_loader, test_balanced_loader, test_imbalanced_loader
    def standard_normal_long_tail(self, cls_num, n_max, imb):
        '''
        cls_num: 类别数
        n_max: 头部样本量
        imb: 不平衡因子
        '''
        # 尾部样本量
        n_min = n_max * imb  

        # 计算尾部位置 x
        x = np.sqrt(-2 * np.log(imb))
        
        # 生成类别索引映射到 [0, x]
        i = np.arange(cls_num)
        x_i = x * i / (cls_num - 1)
        
        # 计算概率密度值
        f_i = np.exp(-x_i**2 / 2) / np.sqrt(2 * np.pi)
        
        # 定义密度值范围
        f_max = 1 / np.sqrt(2 * np.pi)  # 头部密度值
        f_min = imb / np.sqrt(2 * np.pi)  # 尾部密度值
        
        # 线性映射到样本数量
        n_samples = n_min + (f_i - f_min) / (f_max - f_min) * (n_max - n_min)
        n_samples = np.round(n_samples).astype(int)
        
        return np.clip(n_samples, n_min, n_max)
    
    # 根据 samples_per_class 对数据进行上下采样
    def resample_data_by_class(self, data, labels, samples_per_class):
        """根据每个类别的样本数量对数据进行上下采样"""
        class_data = defaultdict(list)

        # 按类别分组数据
        for sample, label in zip(data, labels):
            class_data[label].append(sample)

        resampled_data = []
        resampled_labels = []

        # 对每个类别进行采样
        for class_idx, target_samples in samples_per_class.items():
            if class_idx in class_data:
                current_data = class_data[class_idx]
                if len(current_data) >= target_samples:
                    # 下采样
                    sampled_data = resample(current_data, replace=False, n_samples=target_samples, random_state=42)
                else:
                    # 上采样
                    sampled_data = resample(current_data, replace=True, n_samples=target_samples, random_state=42)
                
                resampled_data.extend(sampled_data)
                resampled_labels.extend([class_idx] * target_samples)

        return np.array(resampled_data), np.array(resampled_labels)
    
    def longtail_process(self, X_train, X_test, y_train, y_test):
        print('=' * 10, 'Long Tail Processing', '=' * 10)
        # 计算每个类别的长尾样本数量
        samples_per_class_train = self.standard_normal_long_tail(self.num_classes, self.n_max, self.imb)
        samples_per_class_train = {i: int(samples) for i, samples in enumerate(samples_per_class_train)}
        print("samples_per_class:", samples_per_class_train)

        data_info_path = self.train_data_path + '/data_info.csv'
        data_info = list(samples_per_class_train.values())
        columns_info = list(str(key) for key in samples_per_class_train.keys())
        if os.path.exists(data_info_path):
            info = pd.read_csv(data_info_path, index_col=0, header=0)
            info = pd.concat([info, pd.DataFrame([data_info],  #index为数据参数
                                            index=['train_{test_size}_{len_window}_{stride}_{type_standard}_snr{snr}_prob{impulse_prob}_amp{impulse_amp}_ratio{imb}_head{n_max}'.format(**self.path_param['train_path_lt'])],
                                            columns=columns_info)],axis=0)
        else:
            info = pd.DataFrame([data_info],
                                index=['train_{test_size}_{len_window}_{stride}_{type_standard}_snr{snr}_prob{impulse_prob}_amp{impulse_amp}_ratio{imb}_head{n_max}'.format(**self.path_param['train_path_lt'])],
                                columns=columns_info)
        info.to_csv(data_info_path, index=True, header=True)

        # 对数据进行上下采样
        X_train, y_train = self.resample_data_by_class(X_train, y_train, samples_per_class_train)
        print("Resampled label distribution:", np.bincount(y_train))

        X_test_balanced, y_test_balanced = self.test_data_process_balanced(X_test, y_test, data_info_path)
        X_test_imbalanced, y_test_imbalanced = self.test_data_process_imbalanced(X_test, y_test, data_info_path)

        return X_train, y_train, X_test_balanced, y_test_balanced, X_test_imbalanced, y_test_imbalanced
        
    def test_data_process_balanced(self, X_test, y_test, data_info_path):
        # 按原测试集各类别中最少数目为均衡测试集的各类数目
        samples_per_class_test = {i: int(samples) for i, samples in enumerate([5000] * 9)}  # [min(np.bincount(y_test))]以最少类别的样本数目作为各类测试样本数
        print("Balanced_samples_per_class:", samples_per_class_test)

        # 各类数目保存到csv表格
        data_info = list(samples_per_class_test.values())
        columns_info = list(str(key) for key in samples_per_class_test.keys())
        info = pd.read_csv(data_info_path, index_col=0, header=0)
        info = pd.concat([info, pd.DataFrame([data_info],  # index为数据参数
                                             index=['test_{test_size}_{len_window}_{stride}_{type_standard}_snr{snr}_prob{impulse_prob}_amp{impulse_amp}'.format(**self.path_param['test_path_lt_balanced'])],
                                             columns=columns_info)], axis=0)
        info.to_csv(data_info_path, index=True, header=True)

        # 按各类数目采样
        X_test, y_test = self.resample_data_by_class(X_test, y_test, samples_per_class_test)
        print("Balanced test label distribution:", np.bincount(y_test))
        return X_test, y_test
    
    def test_data_process_imbalanced(self, X_test, y_test, data_info_path):
        # 计算每个类别的样本数量
        samples_per_class_test = self.standard_normal_long_tail(self.num_classes, self.n_max * self.test_size, self.imb)
        samples_per_class_test = {i: int(samples) for i, samples in enumerate(samples_per_class_test)}
        print("Imbalanced_samples_per_class:", samples_per_class_test)

        #各类数目保存到csv表格
        data_info = list(samples_per_class_test.values())
        columns_info = list(str(key) for key in samples_per_class_test.keys())
        info = pd.read_csv(data_info_path, index_col=0, header=0)
        info = pd.concat([info, pd.DataFrame([data_info],  # index为数据参数
                                             index=['test_{test_size}_{len_window}_{stride}_{type_standard}_snr{snr}_prob{impulse_prob}_amp{impulse_amp}_ratio{imb}_head{n_max}'.format(**self.path_param['test_path_lt_imbalanced'])],
                                             columns=columns_info)], axis=0)
        info.to_csv(data_info_path, index=True, header=True)

        #按各类数目采样
        X_test, y_test = self.resample_data_by_class(X_test, y_test, samples_per_class_test)
        print("Imbalanced test label distribution:", np.bincount(y_test))
        return X_test, y_test
    
    def get_data(self):
        print('='*10, 'Sliding Window Processing', '='*10)
        window_data, window_label = self.get_time_window_from_files()
        print('='*10, 'Done', '='*10)
        print('raw time window shape:', np.array(window_data).shape)
        print('raw label shape:', np.array(window_label).shape)


        # self.save_to_h5(window_data, window_label,
        #                 self.train_data_path + 'raw_window_data_{}_{}.h5'.format(self.len_window, self.stride))

        # 标准化
        print('='*10, 'Normalization and Standardization', '='*10)
        window_data = self.scale_data(window_data, self.type_standard)
        print('='*10, 'Windowed Data Saving', '='*10)
        self.save_to_h5(window_data, window_label,self.train_data_path + '{}_window_data_{}_{}.h5'.format(self.type_standard,self.len_window, self.stride))

        # 分割数据集
        X_train, X_test, y_train, y_test = self.split_data(window_data, window_label)
        del window_data, window_label

        return X_train, X_test, y_train, y_test
    
    def path_decide(self, train_path, test_path, train_path_lt, test_path_lt_balanced, test_path_lt_imbalanced):

        if self.use_longtail:
            decided_train_path = train_path_lt
            decided_test_balanced_path = test_path_lt_balanced
            decided_test_imbalanced_path = test_path_lt_imbalanced
        else:
            decided_train_path = train_path
            decided_test_balanced_path = test_path
            decided_test_imbalanced_path = None

        return decided_train_path, decided_test_balanced_path, decided_test_imbalanced_path
    
    def process(self):
        #路径可随参数变化
        self.path_param = {
            'train_path': {
                'test_size': self.test_size,
                'len_window': self.len_window,
                'stride': self.stride,
                'type_standard': self.type_standard
            },
            'test_path': {
                'test_size': self.test_size,
                'len_window': self.len_window,
                'stride': self.stride,
                'type_standard': self.type_standard
            },
            'train_path_lt': {
                'snr':self.snr,
                'impulse_prob':self.impulse_prob,
                'impulse_amp':self.impulse_amp,
                'test_size':self.test_size,
                'len_window':self.len_window,
                'stride':self.stride,
                'type_standard':self.type_standard,
                'imb':self.imb,
                'n_max':self.n_max
            },
            'test_path_lt_balanced': {
                'snr': self.snr,
                'impulse_prob': self.impulse_prob,
                'impulse_amp': self.impulse_amp,
                'test_size': self.test_size,
                'len_window': self.len_window,
                'stride': self.stride,
                'type_standard': self.type_standard
            },
            'test_path_lt_imbalanced': {
                'snr': self.snr,
                'impulse_prob': self.impulse_prob,
                'impulse_amp': self.impulse_amp,
                'test_size': self.test_size,
                'len_window': self.len_window,
                'stride': self.stride,
                'type_standard': self.type_standard,
                'imb': self.imb,
                'n_max': self.n_max
            }
        }

        #未进行长尾采集的原始训练数据保存路径
        train_path = (self.train_data_path +
                      'train_dataset{test_size}_{len_window}_{stride}_{type_standard}.h5'.format(**self.path_param['train_path']))
        #未进行长尾采集的原始测试数据保存路径
        test_path = (self.train_data_path +
                     'test_dataset{test_size}_{len_window}_{stride}_{type_standard}.h5'.format(**self.path_param['test_path']))
        #长尾采集后的训练数据保存路径
        train_path_lt= (self.train_data_path +
                    'train_dataset{test_size}_{len_window}_{stride}_{type_standard}_snr{snr}_prob{impulse_prob}_amp{impulse_amp}_ratio{imb}_head{n_max}.h5'.format(**self.path_param['train_path_lt']))
        #均衡测试数据保存路径
        test_path_lt_balanced = (self.train_data_path +
                    'test_dataset{test_size}_{len_window}_{stride}_{type_standard}_snr{snr}_prob{impulse_prob}_amp{impulse_amp}.h5'.format(**self.path_param['test_path_lt_balanced']))
        #非均衡测试数据保存路径
        test_path_lt_imbalanced = (self.train_data_path +
                    'test_dataset{test_size}_{len_window}_{stride}_{type_standard}_snr{snr}_prob{impulse_prob}_amp{impulse_amp}_ratio{imb}_head{n_max}.h5'.format(**self.path_param['test_path_lt_imbalanced']))
        #根据是否使用长尾确定实际训练和测试数据路径。只有使用长尾，才有非均衡测试数据路径
        decided_train_path, decided_test_balanced_path, decided_test_imbalanced_path = self.path_decide(train_path, test_path, train_path_lt, test_path_lt_imbalanced, test_path_lt_balanced)

        #首先判断实际训练和测试数据路径是否存在
        if not os.path.exists(decided_train_path) or not os.path.exists(decided_test_balanced_path):
           #其次判断原始的训练和测试数据路径是否存在
           if not os.path.exists(train_path) or not os.path.exists(test_path):
                print('='*10, 'Train Data Path Is Not Exist', '='*10)
                # get_data：按时间窗切数据，并按比例分成原始训练集和测试集
                X_train, X_test, y_train, y_test = self.get_data()
                print('='*10, 'Train Data and Test Data Saving', '='*10)
                self.save_to_h5(X_train, y_train, train_path)
                self.save_to_h5(X_test, y_test, test_path)

           elif os.path.exists(train_path) or os.path.exists(test_path):
                print('='*10, 'Train Data and Test Data Loading', '='*10)
                X_train, y_train = self.load_h5(train_path)
                X_test, y_test = self.load_h5(test_path)
    
           if self.use_longtail:
                print('='*10, 'Using Longtail', '='*10)
                X_train, y_train, X_test_balanced, y_test_balanced, X_test_imbalanced, y_test_imbalanced = self.longtail_process(X_train, X_test, y_train, y_test)
                print('=' * 10, 'Done', '=' * 10)

                if self.add_noise:  #添加噪声
                    print('=' * 10, 'Adding Noise', '=' * 10)
                    X_train = add_mixed_noise(X_train, self.snr, self.impulse_prob, self.impulse_amp)
                    X_test_balanced = add_mixed_noise(X_test_balanced, self.snr, self.impulse_prob, self.impulse_amp)
                    X_test_imbalanced = add_mixed_noise(X_test_imbalanced, self.snr, self.impulse_prob, self.impulse_amp)
                    print('=' * 10, 'Done', '=' * 10)

                print('='*10, 'Train Data LT and Test Data LT Saving', '='*10)
                self.save_to_h5(X_train, y_train, decided_train_path)
                self.save_to_h5(X_test_balanced, y_test_balanced, decided_test_balanced_path)
                self.save_to_h5(X_test_imbalanced, y_test_imbalanced, decided_test_imbalanced_path)

        elif os.path.exists(decided_train_path) or os.path.exists(decided_test_balanced_path):
            print('=' * 10, 'Train Data LT and Test Data LT Loading', '=' * 10)
            X_train, y_train = self.load_h5(decided_train_path)
            X_test_balanced, y_test_balanced = self.load_h5(decided_test_balanced_path)
            if decided_test_imbalanced_path is not None:
                X_test_imbalanced, y_test_imbalanced = self.load_h5(decided_test_imbalanced_path)
            else:
                X_test_imbalanced, y_test_imbalanced = None, None

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test_balanced, y_test_balanced = np.array(X_test_balanced), np.array(y_test_balanced)
        if decided_test_imbalanced_path is not None:
            X_test_imbalanced, y_test_imbalanced = np.array(X_test_imbalanced), np.array(y_test_imbalanced)

        ##构造小样本数据-下采样
        if self.args.down_sampling:
            X_train, y_train = self.manual_undersample(X_train, y_train)
            if self.add_noise:  # 添加噪声
                print('=' * 10, 'Adding Noise', '=' * 10)
                X_train = add_mixed_noise(X_train, self.snr, self.impulse_prob, self.impulse_amp)
                X_test_balanced = add_mixed_noise(X_test_balanced, self.snr, self.impulse_prob, self.impulse_amp)

        train_loader, val_loader, test_balanced_loader, test_imbalanced_loader = self._data_loader(X_train, y_train, X_test_balanced, y_test_balanced, X_test_imbalanced, y_test_imbalanced)

        return train_loader, val_loader, test_balanced_loader, test_imbalanced_loader