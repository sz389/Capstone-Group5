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

# %%
df_image = pd.read_csv('/home/ubuntu/capstone/Wav2Vec2/Sex_Image.csv', index_col = 0)
print(df_image.columns)

# %%
df = df_image[['image_name','path','Sex']]
print(df.head())

# %%
current_path = '/home/ubuntu/capstone/Wav2Vec2/'
df['id'] = current_path + 'Spectograms_Images/'+ df['image_name']
df['path'] = current_path + 'AudioWAV/' + df_image['File Name']
print(df.head())

# %%
xdf = df[['id']]
xdf['label']=df['Sex']
print(xdf.head())

# %%
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
    # if target_type == 2:
    #     ## The target comes as a string  x1, x2, x3,x4
    #     ## the following code creates a list
    #     target = np.array(xdf_data['label'].apply( lambda x : x.split(",")))
    #     final_target = mlb.fit_transform(target)
    #     xfinal = []
    #     if len(final_target) ==0:
    #         xerror = 'Could not process Multilabel'
    #     else:
    #         class_names = mlb.classes_
    #         for i in range(len(final_target)):
    #             joined_string = ",".join( str(e) for e in final_target[i])
    #             xfinal.append(joined_string)
    #         xdf_data['target_class'] = xfinal
    if target_type == 1:
        xtarget = list(np.array(xdf['label'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf['label']))
        class_names=(xtarget)
        xdf['label'] = final_target
    ## We add the column to the main dataset

    return
# %%
process_target(1) #set OUTPUTS_a to # of class names
print(xdf.head())

# %% ----------------------------------- Split data into Train and Validation ----------------------------
save_path = current_path
train_df, test_df = train_test_split(xdf, test_size=0.2, random_state=101, stratify=xdf["label"])

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/Sex_Training/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/Sex_Training/test.csv", sep="\t", encoding="utf-8", index=False)

print(train_df.shape)
print(test_df.shape)

# %%
xdf_dset = train_df
xdf_dset_test = test_df

# %%
import cv2
from torch.utils import data
IMAGE_SIZE = 150
OUTPUTS_a = 2
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

# %%
list_IDS = list(xdf_dset.index)
dataset_obj = Dataset(list_IDS,'train', 1)

list_IDS_test = list(xdf_dset_test.index)
dataset_obj_test = Dataset(list_IDS_test,'test', 1)
BATCH_SIZE = 100

# %%
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

    params = {'batch_size': 1,
              'shuffle': True}
    # transform = T.Compose([T.RandomCrop(size=128,pad_if_needed=True),T.ColorJitter(brightness=0.5),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    training_set = Dataset(partition['train'], 'train', target_type,transform=None)
    print(len(training_set))
    training_generator = data.DataLoader(training_set, **params)
    print(len(training_generator))

    params = {'batch_size': 1,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type,transform=None)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return training_generator, test_generator
    #eturn training_generator
# %%
train_loader, test_loader = read_data(1)
# %%
import torch
import torch.nn as nn
#import torchvision.datasets as dsets
#import torchvision.transforms as transforms
from torch.autograd import Variable
# %%
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.002
IMAGE_SIZE = 150

# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        #
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

# %%
cnn = CNN().to('cuda')
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# %%
# Train the Model
print(len(xdf_dset))
for epoch in range(num_epochs):
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
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(xdf_dset) // batch_size, loss.item()))
# %%
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).to("cuda")
    outputs = cnn(images).detach().to(torch.device('cpu'))
    #print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    #print(predicted)
    total += labels.size(0)
    labels = torch.argmax(labels, dim=1)
    correct += (predicted == labels).sum()

# %%
print(f'Test Accuracy of the model on the {len(xdf_dset_test)} test images: %d %%' % (100 * correct / total))

# %%
# Save the Trained Model
# torch.save(cnn.state_dict(), 'cnn.Sex')

