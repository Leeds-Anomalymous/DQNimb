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
        # 确保输入是 4D 张量 [batch, channel, height, width]
        if len(x.shape) == 3:  
            x = x.unsqueeze(0)
        
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # 展平 + 全连接
        x = x.reshape(x.size(0), -1) 
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        
        return x