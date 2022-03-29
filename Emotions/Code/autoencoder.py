# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('CREMA_D_Image.csv', index_col = 0)

csv_path = '/home/ubuntu/capstone/'
img_path = '/home/ubuntu/capstone/Mel_Spectrograms/'
df = pd.read_csv(csv_path + 'Mel_Spectrograms_1536_512_160.csv')
df['path'] = img_path + df['img_name']

print(df)
#%%
xdf_data1 = df
xdf_data1['id'] = df['path']
xdf_data1['label'] = xdf_data1['emotion']
print(xdf_data1['label'])
print(xdf_data1['id'])

#%%
#Perform Label Encoding on Dataset
from sklearn.preprocessing import LabelEncoder
def process_target(target_type):
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''

    dict_target = {}
    xerror = 0

    if target_type == 1:
        xtarget = list(np.array(xdf_data1['label'].unique()))

        print("xtarget", xtarget)
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data1['label']))
        xtarget.reverse()
        class_names=(xtarget)
        xdf_data1['label'] = final_target

    return class_names


classes=process_target(1) #set OUTPUTS_a to # of class names
print(classes)
print()
print(xdf_data1[['id','img_name','label']])
#%%

#Manually splitting up Train and Test set
xdf_dset, xdf_dset_test = train_test_split(xdf_data1, random_state = 42, test_size = 0.2, stratify = xdf_data1['label'], shuffle = True)

import cv2
from torch.utils import data

IMAGE_SIZE = 128
OUTPUTS_a = 6
class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''

    def __init__(self, list_IDs, type_data, target_type, transform=None):
        # Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type
        self.transform = transform

    def __len__(self):
        # Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        if self.type_data == 'train':
            y = xdf_dset.label.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.label.get(ID)
            if self.target_type == 2:
                y = y.split(",")

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)

            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = xdf_dset.id.get(ID)
        else:
            file = xdf_dset_test.id.get(ID)

        img = cv2.imread(file)
        # sigma = 0.155
        # img = random_noise(img,var = sigma**2)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Augmentation only for train
        X = torch.FloatTensor(img)

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))/255

        return X, y

list_IDS = list(xdf_dset.index)
dataset_obj = Dataset(list_IDS,'train', 1)

list_IDS_test = list(xdf_dset_test.index)
dataset_obj_test = Dataset(list_IDS_test,'test', 1)


#%%
BATCH_SIZE = 64

def read_data(target_type):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file
    ds_inputs = np.array(xdf_dset['id'])
    ds_targets = xdf_dset['label']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)
    #print(list_of_ids)

    # # Datasets
    partition = {
        'train': list_of_ids,
        'test' : list_of_ids_test
    }

    #Data Loaders

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}
    # transform = T.Compose([T.RandomCrop(size=128,pad_if_needed=True),T.ColorJitter(brightness=0.5),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    training_set = Dataset(partition['train'], 'train', target_type,transform=None)
    print("Training Set Length: ", len(training_set))
    training_generator = data.DataLoader(training_set, **params)
    print("Training Generator Length: ", len(training_generator))

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type,transform=None)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator
    #eturn training_generator

train_loader, test_loader = read_data(1)

#%%
#=======================AUTOENCODER==================
# class AutoEncoder1(nn.Module):
#     def __init__(self):
#         super(AutoEncoder1, self).__init__() #256
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 16, (3, 3), padding='same'), #128 * 16
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#
#             nn.Conv2d(16, 32, (3, 3), padding='same'), #128 * 32
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2, 2),#128
#
#             nn.Conv2d(32, 64, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#
#             nn.Conv2d(64, 128, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2, 2),  # 64
#
#             nn.Conv2d(128, 256, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             #nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(256, 256, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(2, 2), # 32
#
#             nn.Conv2d(256, 256, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#
#             nn.Conv2d(256, 256, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(2, 2), #16
#
#             nn.Conv2d(256, 256, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             #nn.MaxPool2d(2, 2),
#             #nn.ReLU()
#
#             nn.Flatten(start_dim=1), #[64,256 ,16, 16]
#
#             nn.Linear(int(256 * 8 * 8), 128),
#
#             nn.Linear(128, 64),
#
#             nn.Linear(64,32),
#             nn.Linear(32,16),
#             nn.Linear(16, 8)
#
#
#         )
#         self.decoder = nn.Sequential(
#
#             nn.Linear(8, 16),
#
#             nn.Linear(16, 32),
#
#             nn.Linear(32, 64),
#
#             nn.Linear(64, 128),
#
#             nn.Linear(128, int(256 * 8 * 8)),
#
#             nn.Unflatten(dim=1, unflattened_size=(256,8,8)), #8
#
#             nn.ConvTranspose2d(256, 256, (3, 3)), #10
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             #nn.Upsample(scale_factor=2),
#
#             nn.ConvTranspose2d(256, 256, (3, 3)), #12
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#
#             nn.ConvTranspose2d(256, 256, (3, 3)), #14
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.Upsample(scale_factor=2),  # 28
#
#             nn.ConvTranspose2d(256, 256, (3, 3)), #30
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#
#             nn.ConvTranspose2d(256, 128, (3, 3)), #32
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor = 2),#64
#
#             nn.ConvTranspose2d(128, 64, (3, 3), padding = (1,1)),#64
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             #nn.Upsample(scale_factor = 2)
#
#             nn.ConvTranspose2d(64, 32, (3, 3), padding = (1,1)),#64
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Upsample(scale_factor=2),#128
#
#             nn.ConvTranspose2d(32, 16, (3, 3), padding = (1,1)),#128
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             #nn.Upsample(scale_factor=2),
#
#             nn.ConvTranspose2d(16, 3, (3, 3), padding = (1,1)),#128
#             nn.ReLU(),
#             nn.BatchNorm2d(3),
#             #nn.Upsample(scale_factor=2)
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, jj , kk):
        super().__init__()
        channels = 3 ; ch1 = 16 ; ch2 = 32 ; ch3 = 64
        kernel_size = (4, 4); padding = (0, 0) ; stride = (2, 2)
        self.enc1 = nn.Conv2d(in_channels=channels, out_channels=ch1, kernel_size=kernel_size,  stride=stride, padding=padding)
        self.relu = nn.ReLU(True)
        self.enc2= nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=kernel_size,  stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(ch2)
        self.relu = nn.ReLU(True)
        self.enc3= nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=kernel_size,  stride=stride, padding=padding)
        self.relu = nn.ReLU(True)
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

def cal(max_sent):
    kernel_size = (4,4); padding = (0,0); dilate = (1,1); maxpool = (1,1); stride = (2,2)

    jj = np.ceil(((max_sent + 2 * padding[0]) - (dilate[0]) * (kernel_size[0] - 1)) / (maxpool[0] * stride[0]))
    kk = np.ceil(((max_sent + 2 * padding[1]) - (dilate[1]) * (kernel_size[1] - 1)) / (maxpool[1] * stride[1]))


    jj = np.ceil(((jj + 2 * padding[0]) - (dilate[0]) * (kernel_size[0] - 1)) / (maxpool[0] * stride[0]))
    kk = np.ceil(((kk + 2 * padding[1]) - (dilate[1]) * (kernel_size[1] - 1)) / (maxpool[1] * stride[1]))


    jj = np.ceil(((jj + 2 * padding[0]) - (dilate[0]) * (kernel_size[0]- 1)) / (maxpool[0] * stride[0]))
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
        self.relu = nn.ReLU(True)
        self.dec2 = nn.ConvTranspose2d(in_channels=ch2, out_channels=ch1, kernel_size=kernel_size, stride=stride, padding=padding1, output_padding=(0,0))
        self.batchnorm2= nn.BatchNorm2d(ch1)
        self.relu = nn.ReLU(True)
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
        return x #return: [64, 3, 128, 128]

# class Encoder(nn.Module):
#     def __init__(self, encoded_space_dim, jj , kk):
#         super().__init__()
#         kernel_size = (4, 4); padding = (0, 0) ; stride = (2, 2)
#
#         self.enc1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size,  stride=stride, padding=padding)
#         self.relu = nn.ReLU(True)
#         self.enc2= nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size,  stride=stride, padding=padding)
#         self.batchnorm = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(True)
#         self.enc3= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size,  stride=stride, padding=padding)
#         self.relu = nn.ReLU(True)
#         self.enc4= nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size,  stride=stride, padding=padding)
#         self.batchnorm2 = nn.BatchNorm2d(128)
#         self.relu = nn.ReLU(True)
#         self.enc5= nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size,  stride=stride, padding=padding)
#         self.relu = nn.ReLU(True)
#
#         self.flatten = nn.Flatten(start_dim=1)
#         jj = 2; kk = 2
#         self.encoder_lin = nn.Sequential(nn.Linear(int(256 * jj *  kk), 128),  nn.Linear(128, encoded_space_dim)) # nn.ReLU(True)
#
#     def forward(self, x):
#         # input: [64, 3, 128, 128]
#         x = self.enc1(x) #[64, 16, 63, 63]
#         x = self.relu(x) #[64, 16, 63, 63]
#         x = self.enc2(x) #[64, 32, 30, 30]
#         x = self.batchnorm(x) #[64, 32, 30, 30]
#         x = self.relu(x) #[64, 32, 30, 30]
#         x = self.enc3(x) #[64, 64, 14, 14]
#         x = self.relu(x) #[64, 64, 14, 14]
#         x = self.enc4(x) #[64, 128, 6, 6]
#         x = self.batchnorm2(x) #[64, 128, 6, 6]
#         x = self.relu(x) #[64, 128, 6, 6]
#         x = self.enc5(x) #[64, 256, 2, 2]
#         x = self.relu(x) #[64, 256, 2, 2]
#         y = self.flatten(x) #[64, 1024]
#         x = self.encoder_lin(y) #1024 > 128 -> 64
#         return x #return: [64]
#
# def cal(max_sent):
#     kernel_size = (4,4); padding = (0,0); dilate = (1,1); maxpool = (1,1); stride = (2,2)
#
#     jj = np.ceil(((max_sent + 2 * padding[0]) - (dilate[0]) * (kernel_size[0] - 1)) / (maxpool[0] * stride[0]))
#     kk = np.ceil(((max_sent + 2 * padding[1]) - (dilate[1]) * (kernel_size[1] - 1)) / (maxpool[1] * stride[1]))
#
#
#     jj = np.ceil(((jj + 2 * padding[0]) - (dilate[0]) * (kernel_size[0] - 1)) / (maxpool[0] * stride[0]))
#     kk = np.ceil(((kk + 2 * padding[1]) - (dilate[1]) * (kernel_size[1] - 1)) / (maxpool[1] * stride[1]))
#
#
#     jj = np.ceil(((jj + 2 * padding[0]) - (dilate[0]) * (kernel_size[0]- 1)) / (maxpool[0] * stride[0]))
#     kk = np.ceil(((kk + 2 * padding[1]) - (dilate[1]) * (kernel_size[1] - 1)) / (maxpool[1] * stride[1]))
#
#     return int(jj) ,int(kk)
#
#
# class Decoder(nn.Module):
#     def __init__(self, encoded_space_dim,jj, kk):
#         super().__init__()
#         kernel_size = (4, 4);  padding = (1, 0);  stride = (2,2); padding1=(0,0)
#         jj=2;kk=2
#         self.decoder_lin = nn.Sequential( nn.Linear(encoded_space_dim, 128),  nn.Linear(128, int(256 * jj * kk))) # nn.ReLU(True)
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, jj, kk))
#
#         self.dec1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding1,  output_padding=(0,0))
#         self.batchnorm1 = nn.BatchNorm2d(128)
#         self.relu = nn.ReLU(True)
#         self.dec2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding1, output_padding=(2,2))
#         self.batchnorm2= nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(True)
#         self.dec3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=kernel_size, stride=stride,padding=padding1, output_padding=padding1)
#         self.batchnorm3= nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(True)
#         self.dec4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=kernel_size, stride=stride,padding=(0,0), output_padding=(0,0))
#         self.batchnorm4= nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(True)
#         self.dec5 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=kernel_size, stride=stride,padding=(0,0), output_padding=(0,0))
#         #self.batchnorm5= nn.BatchNorm2d(3)
#         self.relu = nn.ReLU(True)
#
#
#     def forward(self, x):
#         #input : [64]
#         x = self.decoder_lin(x) #[64, 1024]
#         x = self.unflatten(x) #[64, 256, 2, 2]
#         x = self.dec1(x) #[64, 128, 6, 6]
#         x = self.batchnorm1(x) #[64, 128, 6, 6]
#         x = self.relu(x) #[64, 128, 6, 6]
#         x = self.dec2(x) #[64, 64, 14, 14]
#         x = self.batchnorm2(x) #[64, 64, 14, 14]
#         x = self.relu(x) #[64, 64, 6, 14]
#         x = self.dec3(x) #[64, 32, 30, 30]
#         x = self.batchnorm3(x) #[64, 32, 30, 30]
#         x = self.relu(x) #[64, 32, 30, 30]
#         x = self.dec4(x) #[64, 16, 62, 62]
#         x = self.batchnorm4(x) #[64, 16, 62, 62]
#         x = self.relu(x) #[64, 16, 62, 62]
#         x = self.dec5(x) #[64, 3, 126, 126]
#         x = self.relu(x)
#         return x



steps = []
val_loss = []

model_name = 'autoencoder'
csv_path = os.getcwd()
model_path = csv_path +'/saved_autoencoder/'
if not os.path.exists(model_path):
    os.makedirs(model_name)


print("Starting Autoencoder...")
import numpy as np
import torch
import torch.nn as nn

batch_size = 64
epochs = 200
lr = 1e-3
d = 64

max_sent = IMAGE_SIZE
# jj, kk = cal(max_sent) #jj = 14, kk = 14
#autoencoder = AutoEncoder1().to(device)
jj, kk = cal(max_sent)

encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

# params_to_optimize = [
#     {'params': autoencoder.parameters()}
# ]
optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)
criterion = nn.MSELoss()
for epoch in range(epochs):
    # autoencoder.train()
    encoder.train()
    decoder.train()
    loss = []
    for batch_features, label in train_loader:
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        #encoded_data, decoded_data = autoencoder(batch_features.float())
        encoded_data = encoder(batch_features.float())
        decoded_data = decoder(encoded_data)
        train_loss = criterion(decoded_data, batch_features.float())
        train_loss.backward()
        optimizer.step()
        loss.append(train_loss.detach().cpu().numpy().item())
    losses = np.mean(loss)

    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, losses))

PATH_SAVE = "/home/ubuntu/capstone/saved_autoencoder/"
#torch.save(autoencoder.encoder.state_dict(), PATH_SAVE + 'model_autoencoder_only_encoder_added3linear.pt')
# torch.save(encoder.state_dict(), PATH_SAVE + 'jafari_encoder.pt')
torch.save(decoder.state_dict(), PATH_SAVE + 'jafari_decoder.pt')

temp = decoded_data.detach().to(torch.device('cpu')).numpy()[0].transpose(1,2,0) * 255
cv2.imwrite(PATH_SAVE + 'test_see_model2.jpg',temp)
#======================= END AUTOENCODER =========================


#%% CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN

import torch
import torch.nn as nn
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() #Input: [batch_size, num_channels, height, weidth] = [64, 3, 128, 128]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding='same'),  # 128 * 16
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, (3, 3), padding='same'),  # 128 * 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 128

            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 64

            nn.Conv2d(128, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 32

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 16

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # nn.MaxPool2d(2, 2),
            # nn.ReLU()

            nn.Flatten(start_dim=1),  # [64,256 ,8, 8]
            nn.Linear(int(256 * 8 * 8), 128),
            nn.Linear(128, 64),
            nn.Linear(64,32),
            nn.Linear(32,16),
            nn.Linear(16, 8)
         )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class add_linear(nn.Module):
    def __init__(self, CNN):
        super(add_linear, self).__init__()
        self.model = CNN
        self.classifier = nn.Linear(64,OUTPUTS_a)
    def forward(self,x):
        x = self.model(x)
        x = self.classifier(x)
        return x

#%%
# Hyper Parameters
num_epochs = 100
batch_size = 64 #64
learning_rate = 0.00002
IMAGE_SIZE = 128

PATH_SAVE = "/home/ubuntu/capstone/saved_autoencoder/"
state_dict = torch.load(f=PATH_SAVE+'jafari_encoder.pt')
#"model_autoencoder_only_encoder" = 200 epochs, 9 Convolution, 2 Linear Layers
#"model_autoencoder_only_encoder_added3linear" = 200 epochs, 9 Convolution, 5 Linear Layers

#create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = 'encoder.'+k
    new_state_dict[name] = v


cnn = Encoder(encoded_space_dim=d, jj=jj, kk=kk)
cnn.load_state_dict(state_dict)
cnn = add_linear(cnn)
cnn = cnn.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

#%%
met_best = 0
model_name = 'CNN-9'
csv_path = os.getcwd()
model_path = csv_path +'/saved_cnn_emotion/'
if not os.path.exists(model_path):
   os.makedirs(model_path)

classes = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
patience = 10

print("Starting CNN...")
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        cnn.train()

        images = Variable(images)
        labels = Variable(labels)
        images = images.to('cuda')
        labels = labels.to("cuda")

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(xdf_dset) // batch_size, loss.item()))
    # Test the Model
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    for images, labels in test_loader:
        images = Variable(images).to("cuda")
        outputs = cnn(images).detach().to(torch.device('cpu'))
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted)
        #print(predicted)
        total += labels.size(0)
        labels = torch.argmax(labels, dim=1)
        y_true.extend(labels)
        correct += (predicted == labels).sum()

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classes],
                         columns=[i for i in classes])

    print('Confusion matrix: ')
    print(df_cm)
    print('F1 score: ')
    f1 = f1_score(y_true, y_pred, average = 'weighted')
    print(f1_score(y_true, y_pred, average = 'weighted'))
    print('Precision: ')
    print(precision_score(y_true, y_pred, average = 'weighted'))
    print('Recall: ')
    print(recall_score(y_true, y_pred, average = 'weighted'))
    print('Accuracy: ')
    print(accuracy_score(y_true, y_pred))

    patience = patience - 1

    if f1>met_best:
        patience = 4
        met_best = f1
        epoch_best = epoch
        torch.save(obj =cnn.state_dict(),f=model_path+f'model_{model_name}.pt')
    if patience == 0:
        break
# %%
cnn.load_state_dict(torch.load(f=model_path+f'model_{model_name}.pt'))
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
y_pred = []
y_true = []
for images, labels in test_loader:
    images = Variable(images).to("cuda")
    outputs = cnn(images).detach().to(torch.device('cpu'))
    #print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    y_pred.extend(predicted)
    #print(predicted)
    total += labels.size(0)
    labels = torch.argmax(labels, dim=1)
    y_true.extend(labels)
    correct += (predicted == labels).sum()

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classes],
                     columns=[i for i in classes])
print()
print("Final Scores")
# print()
# print(epoch_best)
print()
print('Confusion matrix: ')
print(df_cm)
print('F1 score: ')
print(f1_score(y_true, y_pred, average = 'weighted'))
print('Precision: ')
print(precision_score(y_true, y_pred, average = 'weighted'))
print('Recall: ')
print(recall_score(y_true, y_pred, average = 'weighted'))
print('Accuracy: ')
print(accuracy_score(y_true, y_pred))
print(f'Test Accuracy of the model on the {len(xdf_dset_test)} test images: %d %%' % (100 * correct / total))
print()
print(f'Distribution of data: ')
for i in range(len(classes)):
    print(f'{classes[i]}: {len(xdf_data1[xdf_data1.label==i])}')