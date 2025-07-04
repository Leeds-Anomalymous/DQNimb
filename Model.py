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
        
        # output_dim计算
        conv_output_size = self._get_conv_output(input_shape)  
        
        # 全连接层
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape) 
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        #print("Conv output size:", x.shape[1:])
        return int(torch.prod(torch.tensor(x.shape[1:]))) # 计算卷积输出维度


    def forward(self, x):
        # 确保输入和模型权重在同一设备
        x = x.to(self.fc1.weight.device)
        
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
            # 如果需要，这里可以根据实际情况重塑数据
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # 展平 + 全连接
        x = x.reshape(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x