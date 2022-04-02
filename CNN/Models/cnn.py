from torch import nn
import torch

class CNN(nn.Module):
    def __init__(self, encoded_space_dim, jj, kk):
        super().__init__()
        channels = 3 ; ch1 = 16 ; ch2 = 32 ; ch3 = 64
        kernel_size = (4, 4); padding = (0, 0) ; stride = (2, 2)
        self.enc1 = nn.Conv2d(in_channels=channels, out_channels=ch1, kernel_size=kernel_size,  stride=stride, padding=padding)
        #self.relu = nn.ReLU(True)
        self.enc2= nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=kernel_size,  stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(ch2)
        #self.relu = nn.ReLU(True)
        self.enc3= nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=kernel_size,  stride=stride, padding=padding)
        #self.relu = nn.ReLU(True)
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(nn.Linear(int(ch3 * jj * kk), 128),  nn.Linear(128, encoded_space_dim)) # nn.ReLU(True)

    def forward(self, x):
        # input: [64, 3, 128, 128]
        x = self.enc1(x)  # output: [64, 16, 63, 63]
        # note: with padding (0,0), height - 1 and width - 1, with padding (1,1) height = 64, width = 64
        # x = self.relu(x)
        x = torch.tanh(x)  # input/output: [64, 16, 63, 63]
        x = self.enc2(x)  # output: [64, 32, 30, 30]
        x = self.batchnorm(x)  # output: [64, 32, 30, 30]
        # x = self.relu(x)
        x = torch.tanh(x)  # output: [64, 32, 30, 30]
        x = self.enc3(x)  # output: [64, 64, 14, 14]
        x = torch.tanh(x)  # output: [64, 64, 14, 14]
        y = self.flatten(x) #output: [64, 12544] = [64, 64 * 14 * 14]
        x = self.encoder_lin(y) #output: [64, 64]
        return x #return: [64, 64]
