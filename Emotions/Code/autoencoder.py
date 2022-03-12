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

df = pd.read_csv('CREMA_D_Image.csv', index_col = 0)
#print(df[['image_name','audio_path','Race']])
#print(df.columns)

df_race = pd.read_csv('CREMA_D_nopath.csv')
df['emotion'] = df_race['emotion']

print("Labels: ")
print()
print(df['emotion'].value_counts())

#df = df[df['Race'] != 'Unknown']
#df['Race'].value_counts()

# %% --------------------------------------- Filter broken and non-existed paths -------------------------------------------------------------------
xdf_data1 = df
path = '/home/ubuntu/capstone/Spectrogram_Images/'
xdf_data1['id'] = path + df["image_name"]
xdf_data1['label'] = xdf_data1['emotion']

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
print(xdf_data1[['id','image_name','label']])
#%%

#Manually splitting up Train and Test set
xdf_dset, xdf_dset_test = train_test_split(xdf_data1, random_state = 42, test_size = 0.2, stratify = xdf_data1['emotion'], shuffle = True)

import cv2
from torch.utils import data

IMAGE_SIZE = 256
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
# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 64
LR = 0.005

#Calculating Sizes
conv_layer = 5
input_length = 128
padding1 = 1
dilate = 1
kernel_size1 = 3
maxpool1 = 2
stride1 = 1
kernel_size2 = kernel_size1
maxpool2 = 2
n_conv_filters = 16

for i in range(conv_layer):
    if i == 0:
        jj = int(((input_length + 2 * padding1) - (dilate) * (kernel_size1 - 1)) / (maxpool1 * stride1))
       #jj = (150 + 2 * 1) - (1) * (3 - 1) / (1 * 1) = 150
    else:
        jj = int(((jj + 2 * padding1) - (dilate) * (kernel_size2 - 1)) / (maxpool2 * stride1))
      # jj = (150 + 2 * 1) - (1) * (3 - 1) / (1 * 1) = 150

    print(jj)

    dimension2_3 = jj * n_conv_filters



class AutoEncoder1(nn.Module):
    def __init__(self):
        super(AutoEncoder1, self).__init__() #256
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding='same'), #128 * 16
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, (3, 3), padding='same'), #128 * 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),#128

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
            #nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2), # 32

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2), #16

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #nn.MaxPool2d(2, 2),
            #nn.ReLU()

        )
        self.decoder = nn.Sequential(

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            #nn.Upsample(scale_factor=2),  # 8

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),  # 16

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, (3, 3), padding = 'same'), #4
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor = 2),#32

            nn.Conv2d(128, 64, (3, 3), padding='same'),#8
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #nn.Upsample(scale_factor = 2) #16

            nn.Conv2d(64, 32, (3, 3), padding='same'),#16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),#64

            nn.Conv2d(32, 16, (3, 3), padding='same'),#32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            #nn.Upsample(scale_factor=2),#64

            nn.Conv2d(16, 3, (3, 3), padding = 'same'),#64
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Upsample(scale_factor=2)#128
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
autoencoder = AutoEncoder1().to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()
loss_func1 = nn.CrossEntropyLoss()

steps = []
val_loss = []

model_name = 'autoencoder'
csv_path = os.getcwd()
model_path = csv_path +'/saved_autoencoder/'
if not os.path.exists(model_path):
    os.makedirs(model_name)

met_best = 1
print("Starting Autoencoder...")
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
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
                met_best = loss.item()
            steps.append(step)
            val_loss.append(loss.item())

#torch.save(obj=autoencoder.encoder.state_dict(), f=model_path + f'model_{model_name}.pt')

#%% CNNNNNNNNNNNNNN

import torch
import torch.nn as nn
from torch.autograd import Variable

# Hyper Parameters
num_epochs = 25
batch_size = BATCH_SIZE #32
learning_rate = 0.00002
IMAGE_SIZE = 150

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Input: [batch_size, channel, height, weidth] = [32, 3, 150, 150]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding='same'),  # 128 * 16
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, (3, 3), padding='same'),  # 128 * 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 64

            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 32

            nn.Conv2d(128, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 16

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 8

            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # nn.MaxPool2d(2, 2),
            nn.ReLU()

        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class add_linear(nn.Module):
    def __init__(self, CNN):
        super(add_linear, self).__init__()
        self.model = CNN
        self.classifier = nn.Linear(256,OUTPUTS_a)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self,x):
        x = self.model(x)
        x = self.classifier(self.global_avg_pool(x).view(-1, 256))
        return x

from torchvision import models
#model = models.resnet18(pretrained=True)
#model = models.resnet34(pretrained=True)
#model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
#model = models.vgg16(pretrained=True)
#model = models.efficientnet_b2(pretrained=True)
#model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,OUTPUTS_a)

#%%
state_dict = torch.load(f=model_path+f'{model_name}.pt')  # 模型可以保存为pth文件，也可以为pt文件。

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = 'encoder.'+k
    new_state_dict[name] = v

cnn = CNN().to(device)
cnn.load_state_dict(new_state_dict)
cnn = add_linear(cnn)
cnn = cnn.to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

#%%
met_best = 0
model_name = 'CNN-9'
csv_path = os.getcwd()
model_path = csv_path +'/saved_cnn_emotion/'
if not os.path.exists(model_path):
   os.makedirs(model_path)

patience = 4
print("Starting CNN...")
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
    print(f'{classes[i]}: {len(xdf_data1[xdf_data1.label==i])}')