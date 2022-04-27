import torch
from torch import nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, jj , kk):
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

def cal(image_size, layers):
    kernel_size = (4,4); padding = (0,0); dilate = (1,1); maxpool = (1,1); stride = (2,2)

    for i in range(layers):
        if i == 0:
            jj = np.ceil(((image_size + 2 * padding[0]) - (dilate[0]) * (kernel_size[0] - 1)) / (maxpool[0] * stride[0]))
            kk = np.ceil(((image_size + 2 * padding[1]) - (dilate[1]) * (kernel_size[1] - 1)) / (maxpool[1] * stride[1]))

        else:
            jj = np.ceil(((jj + 2 * padding[0]) - (dilate[0]) * (kernel_size[0] - 1)) / (maxpool[0] * stride[0]))
            kk = np.ceil(((kk + 2 * padding[1]) - (dilate[1]) * (kernel_size[1] - 1)) / (maxpool[1] * stride[1]))

    return int(jj) ,int(kk)


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim,jj, kk):
        super().__init__()
        channels= 3; ch1= 16; ch2=32; ch3=64;
        kernel_size = (4, 4);  padding = (1, 0);  stride = (2,2); padding1=(0,0)
        self.decoder_lin = nn.Sequential( nn.Linear(encoded_space_dim, 128),  nn.Linear(128, int(ch3 * jj * kk))) # nn.ReLU(True)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(ch3, jj, kk))
        self.dec1 = nn.ConvTranspose2d(in_channels=ch3, out_channels=ch2, kernel_size=kernel_size, stride=stride, padding=padding1,  output_padding=(0,0))
        self.batchnorm1 = nn.BatchNorm2d(ch2)
        #self.relu = nn.ReLU(True)
        self.dec2 = nn.ConvTranspose2d(in_channels=ch2, out_channels=ch1, kernel_size=kernel_size, stride=stride, padding=padding1, output_padding=(1,1))
        self.batchnorm2= nn.BatchNorm2d(ch1)
        #self.relu = nn.ReLU(True)
        self.dec3 = nn.ConvTranspose2d(in_channels=ch1, out_channels=channels, kernel_size=kernel_size, stride=stride,padding=padding1, output_padding=padding1)

    def forward(self, x):
        x = self.decoder_lin(x) ##input: [64,64] output: [64, 12544]
        x = self.unflatten(x) #output: [64, 64, 14, 14]
        # x = self.decoder_conv(x)
        x = self.dec1(x) #output: [64, 32, 30, 30]
        x = self.batchnorm1(x) #output: [64, 32, 30, 30]
        # x = self.relu(x)
        x = torch.tanh(x) #output: [64, 32, 30, 30]
        x = self.dec2(x) #output: [64, 16, 63, 63] #with output padding (0,0) = [64, 16, 62, 62]
        x = self.batchnorm2(x) #output: [64, 16, 63, 63]
        # x = self.relu(x)
        x = torch.tanh(x) #output: [64, 16, 63, 63]
        x = self.dec3(x) #output: [64, 3, 128, 128]
        x = torch.tanh(x) #output: [64, 3, 128, 128]
        return x #return: [64, 3, 128, 128] #

class Classifier(nn.Module):
    def __init__(self, encoder, encoded_space_dim, OUTPUTS_a):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoded_space_dim, OUTPUTS_a)
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

