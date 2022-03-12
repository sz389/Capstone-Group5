#%%
import sys
import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd  # To play sound in the notebook
import librosa
import librosa.display
import tqdm
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
seed = 42

#%%
## ==================== Import Data ====================
import torch.cuda

df_full = pd.read_csv('/home/ubuntu/capstone/CREMA_D_nopath.csv')
df_full['path'] = 'AudioWAV/' + df_full['File Name']
df_full = df_full[df_full['Race'] != 'Unknown']
df_full = df_full[['File Name','path','Race']]



df, df_test = train_test_split(df_full, random_state = 42, test_size = 0.2, stratify = df_full['Race'], shuffle = True)

print("Full Data Ratio: ")
print('Caucasian:',df.Race.value_counts()[0], df.Race.value_counts()[0]/len(df))
print('African American:',df.Race.value_counts()[1], df.Race.value_counts()[1]/len(df))
print('Asian:',df.Race.value_counts()[2], df.Race.value_counts()[2]/len(df))

## ==================== Count Labels ====================
#%%
df_count = df.groupby("Race").count()[["path"]].reset_index()
df_count
Caucasian_num_samples = df_count['path'][2]

print("Caucasian samples: ", Caucasian_num_samples)


#%% ================= Removing Majority Sample ========================
df_count = df_count[df_count['Race'] != 'Caucasian']
df_count = df_count.reset_index(drop=True)
print("DF without Caucasian: \n", df_count)

#%%
## ==================== Calculate Augmentation Times For Minority Lables ====================
import math

num = {}
for i in range(len(df_count)):
  times = math.ceil(Caucasian_num_samples/df_count['path'][i]) # get the augmentation times for each label
  num[df_count['Race'][i]] = times - 1
print("Augmentation times: ", num)

#%%
## ==================== Function To Generate Specturgram ====================#%%
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio
augmented_images_path = '/home/ubuntu/capstone/Augmented_Images/'

#Race
sample_rate = 16000
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 160

#Emotion
# sample_rate = 16000
# n_fft = 1536
# win_length = None
# hop_length = 512
# n_mels = 160
import librosa.feature
def melspec_librosa(x):
    return librosa.feature.melspectrogram(
    y = x,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    n_mels=n_mels,
    norm='slaney',
    htk=True,
)
def save_spectrogram(spec,file,augmentation_times, aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  axs.get_xaxis().set_visible(False)
  axs.get_yaxis().set_visible(False)
  plt.axis('off')
  # plt.show(block=False)
  plt.savefig(
      augmented_images_path + "Augmented_" + str(augmentation_times) + "_" + file + ".jpg",
      bbox_inches='tight', pad_inches=0)

  return file + "_split_" + str(
      augmentation_times) + ".jpg"
#%%
## ==================== Augmentation Part ====================

import random
import librosa
import soundfile as sf
import numpy as np
import torchaudio
import torch

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# path
audio_path = '/home/ubuntu/capstone/AudioWAV/'
augmented_audio_path = '/home/ubuntu/capstone/Augmented_Audio/'
augmented_images_path = '/home/ubuntu/capstone/Augmented_Images/'


# Augmentation methods
def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal

def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(y=signal,rate=time_stretch_rate)

def pitch_shift(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=num_semitones)

def random_gain(signal, min_factor=0.1, max_factor=0.12):
    gain_rate = random.uniform(min_factor, max_factor)
    augmented_signal = signal * gain_rate
    return augmented_signal

def time_shift(signal, sr, factor):
    augmented_signal = np.roll(signal,int(factor)) #int(sr/factor) = 16000/(np.random(15-20))
    return augmented_signal


#%%
# random select augmentation method and factor
from random import shuffle
# funcs = [add_white_noise,time_stretch,pitch_scale, random_gain]
# fun = random.choice(funcs)
# print(fun)
# factor = round(random.uniform(0.4,2), 1)
# print(factor)

def random_method(func1,signal1,sr):
    if func1 == add_white_noise:
        factor = round(random.uniform(0.4, 2), 2)
        return add_white_noise(signal1,factor)
    elif func1 == pitch_shift:
        factor = round(random.uniform(0.4, 2), 2)
        return time_stretch(signal1,factor)
    elif func1 == time_shift:
        factor = round(random.uniform(1,5), 2)
        return time_shift(signal1, sr, factor)
    # factor = round(random.uniform(1,5), 1)
    # return pitch_shift(signal1,sr,factor)
    elif func1 == random_gain:
        return random_gain(signal1,2,4)
    elif func1 == time_stretch:
        factor = round(random.uniform(0.4, 2), 2)



#%%
if __name__ == "__main__":
    Augmented_Audio = pd.DataFrame()  # add augmentation data into this df, then export to csv at the end
    Augmented_Image = pd.DataFrame()  # add augmentation image into this df, then export to csv at the end
    # n = len(df_count['native_langugage'].unique())

    for label in df_count['Race']:  # get label name
        for augmentation_times in range(num[label]):  # get the augmentation times for current label
            #Augmentation times:  {'African American': 2, 'Asian': 8}
            # get augmentation method and factor randomly
            funcs = [time_shift]
            func1 = random.choice(funcs)
            for i in range(len(df[df['Race'] == label])):
                individual_audio_file = df[df['Race'] == label]['File Name'].iloc[i]  # get individual audio file name
                signal1, sr = librosa.load(audio_path + individual_audio_file , sr=16000)  # load individual audio file and resample it to 16000
                #augmented_signal = pitch_scale(signal,sr,factor)  # noise_percentage_factor can be changed here for your preference
                augmented_signal = random_method(func1,signal1,sr)
                #save_graph = generate_mel_spectrogram(augmented_signal, individual_audio_file,augmentation_times)  # generate spectrogram for the augmented audio file
                spec = melspec_librosa(augmented_signal)
                save_graph = save_spectrogram(spec, individual_audio_file, augmentation_times)

                sf.write(augmented_audio_path + "Augmented_" + str(augmentation_times) + "_"
                         + individual_audio_file, augmented_signal, sr)

                Augmented_Audio = Augmented_Audio.append(
                    {'File Name': "Augmented_" + str(augmentation_times) + "_" + individual_audio_file,
                     'label': label}, ignore_index=True)

                Augmented_Image = Augmented_Image.append(
                    {'File Name': 'Augmented_' + str(augmentation_times) + "_" + individual_audio_file + ".jpg"
                    ,'label': label}, ignore_index=True)
                del signal1,augmented_signal,spec,save_graph

        else:
            pass

csv_path = "/home/ubuntu/capstone/"

Augmented_Audio.to_csv(csv_path + 'Augmented_Audio.csv')
Augmented_Image.to_csv(csv_path + 'Augmented_Image.csv')
Augmented_Image.groupby(['label']).count()

#%% ================== Adding Path to Augmented_Audio and Augmented_Image ======================
csv_path = "/home/ubuntu/capstone/"
Augmented_Audio = pd.read_csv(csv_path + "Augmented_Audio.csv")
Augmented_Image = pd.read_csv(csv_path + "Augmented_Image.csv")
Augmented_Audio['path'] = augmented_images_path + Augmented_Audio['File Name']
Augmented_Image['path'] = augmented_images_path + Augmented_Image['File Name']
Augmented_Image['Image_Name'] = Augmented_Image['File Name']

#%%
df['label'] = df['Race']
df_test["label"] = df_test['Race']

df['Image_Name'] = df['File Name'] + "_1024_512_160.jpg"
df_test['Image_Name'] = df_test['File Name'] + "_1024_512_160.jpg"

df['path'] = "/home/ubuntu/capstone/Mel_Spectrograms/" + df['Image_Name']
df_test['path'] = "/home/ubuntu/capstone/Mel_Spectrograms/" + df_test['Image_Name']

augmented_and_original_image_df = pd.concat([Augmented_Image, df[['Image_Name', 'path', 'label']]])
augmented_and_original_image_df = augmented_and_original_image_df.drop(["Unnamed: 0"], axis = 1)
print(augmented_and_original_image_df[['Image_Name', 'path', 'label']])

augmented_and_original_image_df.to_csv('augmented_and_original_image.csv')




# %% CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN CNN

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

df = pd.read_csv("augmented_and_original_image.csv")
print(df.columns)
print()
print(df['label'].value_counts())

xdf_data1 = df.copy()
xdf_data1['id'] = xdf_data1['path']

#%%
print(xdf_data1.columns)

xdf_dset = xdf_data1.copy()

xdf_dset_test = df_test
xdf_dset_test['id'] = xdf_dset_test['path']

print(xdf_dset_test['label'].value_counts())
print(len(xdf_dset_test))
print(xdf_dset_test.columns)

#%%
#Perform Label Encoding on Dataset
from sklearn.preprocessing import LabelEncoder
def process_target():
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''
    # dict_target = {}
    # xerror = 0

    xtarget = list(np.array(xdf_dset['label'].unique()))

    print("xtarget", xtarget)
    le = LabelEncoder()
    le.fit(xtarget)
    final_target = le.transform(np.array(xdf_dset['label']))
    final_target2 = le.transform(np.array(xdf_dset_test['label']))
    #xtarget.reverse()
    class_names=(xtarget)
    xdf_dset['label'] = final_target
    xdf_dset_test['label'] = final_target2

    return class_names
classes = process_target() #set OUTPUTS_a to # of class names
print(classes)

#%%

import cv2
from torch.utils import data

IMAGE_SIZE = 150
OUTPUTS_a = len(classes)

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

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        return X, y


list_IDS = list(xdf_dset.index)
dataset_obj = Dataset(list_IDS,'train', 1)

list_IDS_test = list(xdf_dset_test.index)
dataset_obj_test = Dataset(list_IDS_test,'test', 1)


#%%
BATCH_SIZE = 32

def read_data(target_type):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file
    ds_inputs = np.array(xdf_dset['id'])
    ds_targets = xdf_dset['label']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)
    print(list_of_ids)

    # # Datasets
    partition = {
        'train': list_of_ids,
        'test' : list_of_ids_test
    }

    #Data Loaders

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}
    training_set = Dataset(partition['train'], 'train', target_type,transform=None)
    print(len(training_set))
    training_generator = data.DataLoader(training_set, **params)
    print(len(training_generator))

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type,transform=None)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator
    #eturn training_generator

train_loader, test_loader = read_data(1)

#%%

import torch
import torch.nn as nn
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable

import torch
import torch.nn as nn
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable
# -----------------------------------------------------------------------------------
# Hyper Parameters
num_epochs = 25
batch_size = BATCH_SIZE #32
learning_rate = 0.00002
IMAGE_SIZE = 150


# -----------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.p = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32,64,(3,3))
        self.convnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64,128,(3,3))
        self.convnorm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, (3, 3))
        self.convnorm5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, (3, 3))
        self.convnorm6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 256, (3, 3))
        self.convnorm7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 256, (3, 3))
        self.convnorm8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, (3, 3))
        self.convnorm9 = nn.BatchNorm2d(256)

        # self.conv10 = nn.Conv2d(256, 256, (3, 3))
        # self.convnorm10 = nn.BatchNorm2d(256)
        #
        # self.conv11 = nn.Conv2d(256, 256, (3, 3))
        # self.convnorm11 = nn.BatchNorm2d(256)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(256, OUTPUTS_a)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.p(self.pad1(self.convnorm2(self.act(self.conv2(x)))))
        x = self.pad1(self.convnorm3(self.act(self.conv3(x))))
        x = self.p(self.pad1(self.convnorm4(self.act(self.conv4(x)))))
        x = self.pad1(self.convnorm5(self.act(self.conv5(x))))
        x = self.p(self.pad1(self.convnorm6(self.act(self.conv6(x)))))
        x = self.pad1(self.convnorm7(self.act(self.conv7(x))))
        x = self.p(self.pad1(self.convnorm8(self.act(self.conv8(x)))))
        x = self.act(self.convnorm9(self.act(self.conv9(x))))
        # x = self.p(self.pad1(self.convnorm10(self.act(self.conv10(x)))))
        # x = self.act(self.convnorm11(self.act(self.conv11(x))))

        return self.linear(self.global_avg_pool(x).view(-1, 256))

# -----------------------------------------------------------------------------------
cnn = CNN().to("cuda")
from torchvision import models
#model = models.resnet18(pretrained=True)
#model = models.resnet34(pretrained=True)
#model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
#model = models.vgg16(pretrained=True)
#model = models.efficientnet_b2(pretrained=True)
#model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,OUTPUTS_a)
#cnn = model.to('cuda')
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model
print("Length of xdf_dset: ", len(xdf_dset))

print("Starting CNN...")
met_best = 0
model_name = 'CNN-9'
csv_path = os.getcwd()
model_path = csv_path +'/saved_cnn_race_aug/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
patience = 4
classes = ("African American",'Asian','Caucasian')
#classes = ('ANG','DIS','FEA','HAP','SAD',)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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
        if (i + 1) % 100 == 0:
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

# Constant for classes
#     classes = process_target(1)
    # Build confusion matrix
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
classes = ("African American",'Asian','Caucasian')
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

# Constant for classes
#     classes = process_target(1)
# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classes],
                     columns=[i for i in classes])
print()
print("Final Scores")
print()
print(epoch_best)
print()
print('Confusion matrix: ')
print(df_cm)
print('F1 score: ')
print(met_best)
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
    print(f'{classes[i]}: {len(xdf_data1[xdf_dset.label==i])}')