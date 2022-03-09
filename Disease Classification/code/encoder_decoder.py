#%%
# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from torchvision import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import classification_report
import pydub
import librosa
import soundfile
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils import data
import cv2
import matplotlib.pyplot as plt
#%%
os.chdir('..')
csv_path = os.getcwd()+'/26-29_09_2017_KCL/'
model_path = csv_path +'saved_cnn/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
#%%
# xdf_data1 = pd.read_csv(csv_path+'KCL_spec.csv')
xdf_data1 = pd.read_csv(csv_path+'1024_512_128_KCL_spec.csv')
#%%
xdf_data1 = xdf_data1[['label','img_name']]
xdf_data1.columns =['label','id']
# xdf_data1['id'] = csv_path+'mel_spectrograms/'+xdf_data1['id']
xdf_data1['id'] = csv_path+'try_image/'+xdf_data1['id']
#%%
# Hyper Parameters
num_epochs = 15
BATCH_SIZE = 16
learning_rate = 0.001
from sklearn.preprocessing import LabelEncoder
def process_target():
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''
    xtarget = list(np.array(xdf_data1['label'].unique()))
    le = LabelEncoder()
    le.fit(xtarget)
    final_target = le.transform(np.array(xdf_data1['label']))
    xtarget.reverse()
    class_names=xtarget
    xdf_data1['label'] = final_target
    return class_names
#%%
IMAGE_SIZE=128
target_sampling_rate = 16000
class_names = process_target()
OUTPUTS_a = len(class_names)
class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''

    def __init__(self, df, transform):
        # Initialization'
        self.df = df
        # self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        # Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        # Generates one sample of data'
        # Select sample
        # Load data and get label
        y=self.df.label.iloc[index]
        labels_ohe = np.zeros(OUTPUTS_a)
        for idx, label in enumerate(range(OUTPUTS_a)):
            if label == y:
                labels_ohe[idx] = 1
        y = torch.FloatTensor(labels_ohe)
        file_name = self.df.id.iloc[index]
        img = cv2.imread(file_name)
        # sigma = 0.155
        # img = random_noise(img,var = sigma**2)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        X = torch.FloatTensor(img)
        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))/255.
        return X, y
#%%
def read_data():
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file
    # ---------------------- Parameters for the data loader --------------------------------

    #Data Loaders
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}
    training_set = Dataset(xdf_dset,transform=None)
    training_generator = data.DataLoader(training_set, **params)
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}
    test_set = Dataset(xdf_dset_test,transform=None)
    test_generator = data.DataLoader(test_set, **params)
    return training_generator, test_generator

xdf_dset, xdf_dset_test = train_test_split(xdf_data1, test_size=0.2, random_state=101, stratify=xdf_data1["label"])
train_loader, test_loader = read_data()
#%%
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 128),  # nn.Linear(28*28, 128)
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 3 * IMAGE_SIZE * IMAGE_SIZE),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
class autoencoder1(nn.Module):
    def __init__(self):
        super(autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.AdaptiveMaxPool2d(64),

            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # nn.AdaptiveMaxPool2d(32),

            nn.Conv2d(128, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # nn.AdaptiveMaxPool2d(64),

            nn.Conv2d(128, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.AdaptiveMaxPool2d(128),

            nn.Conv2d(32, 3, (3, 3), padding='same'),
            nn.ReLU(),
            # nn.BatchNorm2d(128),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
autoencoder = autoencoder1().to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()
loss_func1 = nn.CrossEntropyLoss()
steps = []
val_loss = []
model_name = 'encoder'
met_best = 1
for epoch in range(num_epochs):
    for step, (x, y) in enumerate(train_loader):
        #print(x.shape) #torch.Size([32, 3, 150, 150])
        #print()
        #print(y.shape) #torch.Size([32, 3])
        b_x = Variable(x)  # batch x, shape (batch, 28*28) .view(32, 3*150*150)
        b_y = Variable(x)  # batch y, shape (batch, 28*28) .view(32, 3*150*150)
        b_label = Variable(y)               # batch label

        b_x = b_x.to(device)
        b_y = b_y.to(device)
        b_label = b_label.to(device)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if (step+1) % ((len(xdf_dset) // BATCH_SIZE)//4) == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
            if met_best > loss.item():
                torch.save(autoencoder.encoder.state_dict(),model_path+f'{model_name}.pt')
            steps.append(step)
            val_loss.append(loss.item())
#%%
temp = decoded.detach().to(torch.device('cpu')).numpy()[2].transpose(1,2,0) *255
cv2.imwrite(csv_path+'test_see_model2.jpg',temp)