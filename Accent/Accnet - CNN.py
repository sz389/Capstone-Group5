


# %% --------------------------------------- Reading in CSV File with Labels -------------------------------------------------------------------
import pandas as pd
csv_path = '/home/ubuntu/Capstone/'
df = pd.read_csv(csv_path + 'New Accent_Archive_Clean.csv', index_col = 0)
df = df[['filename', 'path', 'native_language']]
# only use top 10
top10 = ['english','spanish','arabic','mandarin','french',
         'korean','portuguese','russian','dutch','turkish']
df1 = df[df['native_language'].isin(top10)]
df = df1
df = df.drop('path', 1)
df['path'] = '/home/ubuntu/Capstone/recordings/recordings/'+ df['filename'] + '.mp3'
df['image_name'] = df['filename'] + ".jpg"
df.head()
df.to_csv("/home/ubuntu/Capstone/Accent_Image.csv")

# %% --------------------------------------- Filter broken and non-existed paths -------------------------------------------------------------------
print(f"Step 0: {len(df)}")
import os

df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
df = df.dropna(subset=["path"])
df = df.drop("status", 1)
print(f"Step 1: {len(df)}")

df = df.sample(frac=1)
df = df.reset_index(drop=True)
df.head()


# %% ------------------------------------- Print our unique labels -----------------------------------------------------
print("Labels: ", df["native_language"].unique())
print()
df.groupby("native_language").count()[["path"]]

import matplotlib.pyplot as plt
import torch


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')

    # figure.suptitle(title)
    # plt.show(block=False)
    return figure


import torchaudio

def extract_waveform_and_sample_rate(df):
    image_list = []
    path_list = []

    i = 0
    for path in df['path']:
        waveform, sample_rate = torchaudio.load(path)  # Ca
        figure = plot_specgram(waveform, sample_rate)
        file_name = df["filename"][i]
        plt.savefig(
            f"/home/ubuntu/Capstone/Spectrogram/{file_name}.jpg",
            bbox_inches='tight', pad_inches=0)
        path_list.append(
            f"/home/ubuntu/Capstone/Spectrogram/{file_name}.jpg")
        i = i + 1
    return image_list, path_list

# %% --------------------------------------- Generate Spectrogram Images -------------------------------------------------------------------
# image_list, path_list = extract_waveform_and_sample_rate(df)

# %% --------------------------------------- Import Packages -------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# %% --------------------------------------- CNN Starts From Here-------------------------------------------------------------------
image_path = '/home/ubuntu/Capstone/'
df_image = pd.read_csv(image_path + 'Accent_Image.csv', index_col = 0)
csv_path = '/home/ubuntu/Capstone/'

df_image = df_image.sample(frac=1)
df_image = df_image.reset_index(drop=True)
print(df_image.head())

xdf_data1 = df_image
path = "/home/ubuntu/Capstone/Spectrogram/"
xdf_data1['id'] = path + df_image["image_name"]
xdf_data1['label'] = xdf_data1['native_language']
print(xdf_data1.head())


# %% ------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer as mlb
def process_target(target_type):
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''

    dict_target = {}
    xerror = 0

    #if target_type == 2:
    #     ## The target comes as a string  x1, x2, x3,x4
    #     ## the following code creates a list
    #     target = np.array(xdf_data1['label'].apply( lambda x : x.split(",")))
    #     final_target = mlb.fit_transform(target)
    #     xfinal = []
    #     if len(final_target) ==0:
    #         xerror = 'Could not process Multilabel'
    #     else:
    #         class_names = mlb.classes_
    #         for i in range(len(final_target)):
    #             joined_string = ",".join( str(e) for e in final_target[i])
    #             xfinal.append(joined_string)
    #         xdf_data1['target_class'] = xfinal

    if target_type == 1:
        xtarget = list(np.array(xdf_data1['label'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data1['label']))
        class_names=(xtarget)
        xdf_data1['label'] = final_target

    ## We add the column to the main dataset


    return class_names

# multi-classification
process_target(1)

# %% ------------------------------- Splitting Data ------------------------------------------------
xdf_dset = xdf_data1[0:962]
xdf_dset_test = xdf_data1[963:]

# %% ------------------------------------------------------------------------------------------------------------------------------------------------
import cv2
from torch.utils import data

OUTPUTS_a = 10 ###### Number of labels
IMAGE_SIZE = 150

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
            y = xdf_dset.label.get(ID)   ########## check if to use train_dataset here
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.label.get(ID)   ########## check if to use test_dataset here
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
            file = xdf_dset.id.get(ID)    ########## check if to use train_dataset here
        else:
            file = xdf_dset_test.id.get(ID)    ########## check if to use test_dataset here

        img = cv2.imread(file)
        # sigma = 0.155
        # img = random_noise(img,var = sigma**2)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Augmentation only for train
        X = torch.FloatTensor(img)

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        return X, y


# %% ---------------------------------------------------------------------------------------------------------------------------------------------------
list_IDS = list(xdf_dset.index)
dataset_obj = Dataset(list_IDS,'train', 1)

list_IDS_test = list(xdf_dset_test.index)
dataset_obj_test = Dataset(list_IDS_test,'test', 1)
BATCH_SIZE = 100
# %% ---------------------------------------------------------------------------------------------------------------------------------------------------
def read_data(target_type):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file

    ds_inputs = np.array(xdf_dset['id'])
    ds_targets = xdf_dset['label']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)


    # # Datasets
    partition = {
        'train': list_of_ids,
        'test' : list_of_ids_test
    }

    #Data Loaders

    params = {'batch_size':  BATCH_SIZE,
              'shuffle': True}
    #transform = T.Compose([T.RandomCrop(size=128,pad_if_needed=True),T.ColorJitter(brightness=0.5),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    training_set = Dataset(partition['train'], 'train', target_type,transform=None)
    training_generator = data.DataLoader(training_set, **params)
    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type,transform=None)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator

train_loader, test_loader = read_data(1)

# %% ----------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable
# %% ----------------------------------------------------- Parameters -----------------------------------------------------------
# Hyper Parameters
num_epochs = 5
batch_size = 10
learning_rate = 0.01
IMAGE_SIZE = 150

# %% ------------------------------------------------------ CNN Model ------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32,64,(3,3))
        self.convnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64,128,(3,3))
        self.convnorm4 = nn.BatchNorm2d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, OUTPUTS_a)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pad1(self.convnorm2(self.act(self.conv2(x))))
        # x = self.act(self.conv2(self.act(x)))
        x = self.pad1(self.convnorm3(self.act(self.conv3(x))))
        x = self.act(self.convnorm4(self.act(self.conv4(x))))
        return self.linear(self.global_avg_pool(x).view(-1, 128))
# %% ----------------------------------------------------------------------
cnn = CNN()
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model
print("Training Starts...")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(xdf_dset) // batch_size, loss.item()))
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    #print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    #print(predicted)
    print()
    total += labels.size(0)
    labels = torch.argmax(labels, dim=1)
    correct += (predicted == labels).sum()
#-----------------------------------------------------------------------------------
print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')



