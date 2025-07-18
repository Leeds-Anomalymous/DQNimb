import torch
import torch.nn as nn

class Q_Net_image(nn.Module):
    def __init__(self, input_shape, output_dim=10): 
        super(Q_Net_image, self).__init__()
        channels, height, width = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 先不初始化全连接层，在第一次前向传播时动态创建
        self.fc1 = None
        self.relu3 = nn.ReLU()
        self.fc2 = None
        self.output_dim = output_dim

    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape) 
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return int(torch.prod(torch.tensor(x.shape[1:]))) # 计算卷积输出维度

    def _initialize_fc_layers(self, conv_output_size):
        """动态初始化全连接层"""
        if self.fc1 is None:
            self.fc1 = nn.Linear(conv_output_size, 256)
            self.fc2 = nn.Linear(256, self.output_dim)
            # 确保新层在正确的设备上
            device = next(self.parameters()).device
            self.fc1 = self.fc1.to(device)
            self.fc2 = self.fc2.to(device)

    def forward(self, x):
        # 确保输入和模型权重在同一设备
        x = x.to(next(self.parameters()).device)
        
        # 根据输入形状判断数据类型并进行相应处理
        if len(x.shape) == 3 and x.shape[1] > 1 and x.shape[2] > 1:
            # TBM数据形状为 [batch_size, len_window, feature_dim]
            # 需要转换为 [batch_size, channels, len_window, feature_dim]
            x = x.unsqueeze(1)  # 添加通道维度 [batch_size, 1, len_window, feature_dim]
            x = x.repeat(1, 3, 1, 1)  # 复制为3通道 [batch_size, 3, len_window, feature_dim]
        elif len(x.shape) == 3:  
            # MNIST/CIFAR10数据 - 单个样本时
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            # 扁平化的输入数据，重塑为所需的形状
            x = x.unsqueeze(0).unsqueeze(0)  # [batch_size, 1, height*width]
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        # 智能池化策略：根据维度大小选择合适的池化方式
        if x.shape[2] < 2 or x.shape[3] < 2:
            # 维度太小时使用自适应池化，保证至少有1个输出
            target_h = max(1, x.shape[2] // 2)
            target_w = max(1, x.shape[3] // 2)
            x = nn.AdaptiveMaxPool2d((target_h, target_w))(x)
        else:
            # 维度足够时使用普通池化（效率更高）
            x = self.pool1(x)
            
        x = self.conv2(x)
        x = self.relu2(x)
        
        # 第二次池化使用相同策略
        if x.shape[2] < 2 or x.shape[3] < 2:
            target_h = max(1, x.shape[2] // 2)
            target_w = max(1, x.shape[3] // 2)
            x = nn.AdaptiveMaxPool2d((target_h, target_w))(x)
        else:
            x = self.pool2(x)
            
        # 展平 + 全连接
        x = x.reshape(x.size(0), -1)
        
        # 动态初始化全连接层
        if self.fc1 is None:
            self._initialize_fc_layers(x.shape[1])
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_1layer(nn.Module):
    """单层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_1layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # 只有一层卷积
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 只经过一次池化
        conv_output_size = (len_window // 2) * 32
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 单个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d(nn.Module):
    """双层卷积的TBM模型 - 用于参数敏感性分析（原始版本）"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        ''''output_length = (1024 + 2*2 - 5) // 1 + 1
              = (1028 - 5) // 1 + 1
              = 1023 + 1
              = 1024'''
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小
        conv_output_size = (len_window // 4) * 32  # 经过两次池化层，长度变为原来的1/4
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:  # 如果输入是[batch, len_window, feature_dim]
            x = x.transpose(1, 2)  # 转换为[batch, feature_dim, len_window]
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_3layer(nn.Module):
    """三层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_3layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 经过三次池化
        conv_output_size = (len_window // 8) * 64
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x

class TBM_conv1d_4layer(nn.Module):
    """四层卷积的TBM模型 - 用于参数敏感性分析"""
    def __init__(self, input_shape, output_dim=2): 
        super(TBM_conv1d_4layer, self).__init__()
        len_window, feature_dim = input_shape 
        
        # Layer 1
        self.conv1 = nn.Conv1d(feature_dim, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Layer 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积输出大小 - 经过四次池化
        conv_output_size = (len_window // 16) * 64
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # 确保输入形状为 [batch_size, feature_dim, len_window]
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # 扁平化
        x = x.flatten(1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        
        return x