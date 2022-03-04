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
# %% --------------------------------------- Reading in CSV File with Labels -------------------------------------------------------------------
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
    # le = LabelEncoder()
    # le.fit(xtarget)
    # final_target = le.transform(np.array(xdf_data1['label']))
    # xtarget.reverse()
    for i, label in enumerate(xtarget):
        xdf_data1['label'].replace(label,i,inplace=True)
    class_names=xtarget
    # xdf_data1['label'] = final_target
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
        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))
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

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(256, OUTPUTS_a)
        self.act = torch.relu
    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.p(self.pad1(self.convnorm2(self.act(self.conv2(x)))))
        x = self.pad1(self.convnorm3(self.act(self.conv3(x))))
        x = self.p(self.pad1(self.convnorm4(self.act(self.conv4(x)))))
        x = self.act(self.convnorm5(self.act(self.conv5(x))))
        x = self.p(self.pad1(self.convnorm6(self.act(self.conv6(x)))))
        x = self.pad1(self.convnorm7(self.act(self.conv7(x))))
        x = self.p(self.pad1(self.convnorm8(self.act(self.conv8(x)))))
        x = self.act(self.convnorm9(self.act(self.conv9(x))))

        return self.linear(self.global_avg_pool(x).view(-1, 256))
#%%
model=CNN()
mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
from torchvision import models
model_name='cnn'
# model = models.resnet18(pretrained=True)
# model = models.resnet34(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.efficientnet_b2(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
# model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,OUTPUTS_a)
cnn = model.to(device)
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#%%
# Train the Model
def train_test():
    met_best = 0
    for epoch in range(num_epochs):
        cnn.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            images = images.to('cuda')
            labels = labels.to('cuda')

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % ((len(xdf_dset) // BATCH_SIZE)//2) == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(xdf_dset) // BATCH_SIZE, loss.item()))
    # %%
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
        # classes = ('Female','Male')

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) *2 , index=[i for i in class_names],
                             columns=[i for i in class_names])
        print('classification report: ')
        print(classification_report(y_true, y_pred, target_names=class_names))
        print('Confusion matrix: ')
        print(df_cm)
        f1 =f1_score(y_true, y_pred)
        print(f'F1 score: {f1}')
        print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
        if f1>met_best:
            met_best=f1
            torch.save(cnn.state_dict(), model_path+"model_{}.pt".format(model_name))
            print('best model saved')
#%%
train_test()
print('The result from the best model:')
cnn.load_state_dict(torch.load(model_path+"model_{}.pt".format(model_name)))
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
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) *2 , index=[i for i in class_names],
                     columns=[i for i in class_names])
print('classification report: ')
print(classification_report(y_true, y_pred, target_names=class_names))
print('Confusion matrix: ')
print(df_cm)
f1 =f1_score(y_true, y_pred)
print(f'F1 score: {f1}')
print(f'Accuracy: {accuracy_score(y_true, y_pred)}')

print(f'Distribution of data: ')
for i in range(len(class_names)):
    print(f'{class_names[i]}: {len(xdf_data1[xdf_data1.label==i])}')