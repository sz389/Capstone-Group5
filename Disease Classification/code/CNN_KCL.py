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
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

# %% --------------------------------------- Reading in CSV File with Labels -------------------------------------------------------------------
os.chdir('..')
csv_path = os.getcwd()+'/26-29_09_2017_KCL/'
# image_path = csv_path+'Spectrogram_Images/'
image_path = csv_path+'Spectrogram_Images_split/'
xdf_data1 = pd.read_csv(csv_path+'KCL_Image.csv')
xdf_data1 = pd.read_csv(csv_path+'KCL_Image_split.csv')
xdf_data1 = xdf_data1[['Disease','image_name']]
xdf_data1.columns =['label','id']
xdf_data1['id'] = image_path+xdf_data1['id']
# df_train = pd.read_csv(csv_path + 'KCL_Image_train.csv',sep='\t')
# df_test = pd.read_csv(csv_path + 'KCL_Image_test.csv',sep='\t')
# df_train = df_train[['Disease','image_name']]
# df_test = df_test[['Disease','image_name']]
#%%
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
# -----------------------------------------------------------------------------------
# Hyper Parameters
num_epochs = 20
n_epoch = 20
BATCH_SIZE = 33
learning_rate = 0.001

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True
NICKNAME = 'Zeqiu'
#%%
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
        xtarget = list(np.array(xdf_data1['label'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data1['label']))
        class_names=(xtarget)
        xdf_data1['label'] = final_target

    ## We add the column to the main dataset


    return class_names
#%%
IMAGE_SIZE=50
#%%
import cv2
from torch.utils import data


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
#%%
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

    params = {'batch_size': BATCH_SIZE,
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
#%%
class_names = process_target(1)
OUTPUTS_a = len(class_names)
#%%
xdf_dset, xdf_dset_test = train_test_split(xdf_data1, test_size=0.2, random_state=101, stratify=xdf_data1["label"])
train_loader, test_loader = read_data(1)
#%%
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
#%%
from torchvision import models

# model = models.resnet18(pretrained=True)
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
#%%
# -----------------------------------------------------------------------------------
# cnn = CNN().to('cuda')
cnn = model.to(device)
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#%%
# Train the Model
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

        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(xdf_dset) // BATCH_SIZE, loss.item()))
#%%
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).to('cuda')
    outputs = cnn(images).detach().to(torch.device('cpu'))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    labels = torch.argmax(labels, dim=1)
    correct += (predicted == labels).sum()
#-----------------------------------------------------------------------------------
print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')
#%%
# def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on,
#                    # pretrained = False,
#                    ):
#     # Use a breakpoint in the code line below to debug your script.
#
#     # model, optimizer, criterion, scheduler = model_definition(pretrained)
#     model = cnn
#     cont = 0
#     train_loss_item = list([])
#     test_loss_item = list([])
#
#     pred_labels_per_hist = list([])
#
#     model.phase = 0
#
#     met_test_best = 0
#     for epoch in range(n_epoch):
#         train_loss, steps_train = 0, 0
#
#         model.train()
#
#         pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#         train_hist = list([])
#         test_hist = list([])
#
#         with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:
#
#             for xdata,xtarget in train_ds:
#
#                 xdata, xtarget = xdata.to(device), xtarget.to(device)
#
#                 optimizer.zero_grad()
#
#                 # output = model(xdata,xdata)
#                 output = model(xdata)
#                 loss = criterion(output, xtarget)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
#                 cont += 1
#
#                 steps_train += 1
#
#                 train_loss_item.append([epoch, loss.item()])
#
#                 pred_labels_per = output.detach().to(torch.device('cpu')).numpy()
#
#                 if len(pred_labels_per_hist) == 0:
#                     pred_labels_per_hist = pred_labels_per
#                 else:
#                     pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])
#
#                 if len(train_hist) == 0:
#                     train_hist = xtarget.cpu().numpy()
#                 else:
#                     train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])
#
#                 pbar.update(1)
#                 pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))
#
#                 pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
#                 #real_labels = np.vstack((real_labels, xtarget.numpy()))
#                 real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
#
#         pred_labels = pred_logits[1:]
#         pred_labels[pred_labels >= THRESHOLD] = 1
#         pred_labels[pred_labels < THRESHOLD] = 0
#
#         # Metric Evaluation
#         train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
#
#         avg_train_loss = train_loss / steps_train
#
#         ## Finish with Training
#
#         ## Testing the model
#
#         model.eval()
#
#         pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#         test_loss, steps_test = 0, 0
#         met_test = 0
#
#         with torch.no_grad():
#
#             with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:
#
#                 for xdata,xtarget in test_ds:
#
#                     xdata, xtarget = xdata.to(device), xtarget.to(device)
#
#                     optimizer.zero_grad()
#
#                     output = model(xdata)
#                     # output = model(xdata,xdata)
#                     loss = criterion(output, xtarget)
#
#                     test_loss += loss.item()
#                     cont += 1
#
#                     steps_test += 1
#
#                     test_loss_item.append([epoch, loss.item()])
#
#                     pred_labels_per = output.detach().to(torch.device('cpu')).numpy()
#
#                     if len(pred_labels_per_hist) == 0:
#                         pred_labels_per_hist = pred_labels_per
#                     else:
#                         pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])
#
#                     if len(test_hist) == 0:
#                         tast_hist = xtarget.cpu().numpy()
#                     else:
#                         test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])
#
#                     pbar.update(1)
#                     pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))
#
#                     pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
#                     # real_labels = np.vstack((real_labels, xtarget.numpy()))
#                     real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
#
#         pred_labels = pred_logits[1:]
#         pred_labels[pred_labels >= THRESHOLD] = 1
#         pred_labels[pred_labels < THRESHOLD] = 0
#
#         test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
#
#         #acc_test = accuracy_score(real_labels[1:], pred_labels)
#         #hml_test = hamming_loss(real_labels[1:], pred_labels)
#
#         avg_test_loss = test_loss / steps_test
#
#         xstrres = "Epoch {}: ".format(epoch)
#         for met, dat in train_metrics.items():
#             xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)
#
#
#         xstrres = xstrres + " - "
#         for met, dat in test_metrics.items():
#             xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
#             if met == save_on:
#                 met_test = dat
#
#         print(xstrres)
#
#         if met_test > met_test_best and SAVE_MODEL:
#
#             torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
#             xdf_dset_results = xdf_dset_test.copy()
#
#             ## The following code creates a string to be saved as 1,2,3,3,
#             ## This code will be used to validate the model
#             xfinal_pred_labels = []
#             for i in range(len(pred_labels)):
#                 joined_string = ",".join(str(int(e)) for e in pred_labels[i])
#                 xfinal_pred_labels.append(joined_string)
#
#             xdf_dset_results['results'] = xfinal_pred_labels
#
#             xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index = False)
#             print("The model has been saved!")
#             met_test_best = met_test
#
#
# def metrics_func(metrics, aggregates, y_true, y_pred):
#     '''
#     multiple functiosn of metrics to call each function
#     f1, cohen, accuracy, mattews correlation
#     list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
#     list of aggregates : avg, sum
#     :return:
#     '''
#
#     def f1_score_metric(y_true, y_pred, type):
#         '''
#             type = micro,macro,weighted,samples
#         :param y_true:
#         :param y_pred:
#         :param average:
#         :return: res
#         '''
#         res = f1_score(y_true, y_pred, average=type)
#         return res
#
#     def cohen_kappa_metric(y_true, y_pred):
#         res = cohen_kappa_score(y_true, y_pred)
#         return res
#
#     def accuracy_metric(y_true, y_pred):
#         res = accuracy_score(y_true, y_pred)
#         return res
#
#     def matthews_metric(y_true, y_pred):
#         res = matthews_corrcoef(y_true, y_pred)
#         return res
#
#     def hamming_metric(y_true, y_pred):
#         res = hamming_loss(y_true, y_pred)
#         return res
#
#     xcont = 1
#     xsum = 0
#     xavg = 0
#     res_dict = {}
#     for xm in metrics:
#         if xm == 'f1_micro':
#             # f1 score average = micro
#             xmet = f1_score_metric(y_true, y_pred, 'micro')
#         elif xm == 'f1_macro':
#             # f1 score average = macro
#             xmet = f1_score_metric(y_true, y_pred, 'macro')
#         elif xm == 'f1_weighted':
#             # f1 score average =
#             xmet = f1_score_metric(y_true, y_pred, 'weighted')
#         elif xm == 'coh':
#              # Cohen kappa
#             xmet = cohen_kappa_metric(y_true, y_pred)
#         elif xm == 'acc':
#             # Accuracy
#             xmet =accuracy_metric(y_true, y_pred)
#         elif xm == 'mat':
#             # Matthews
#             xmet =matthews_metric(y_true, y_pred)
#         elif xm == 'hlm':
#             xmet =hamming_metric(y_true, y_pred)
#         else:
#             xmet = 0
#
#         res_dict[xm] = xmet
#
#         xsum = xsum + xmet
#         xcont = xcont +1
#
#     if 'sum' in aggregates:
#         res_dict['sum'] = xsum
#     if 'avg' in aggregates and xcont > 0:
#         res_dict['avg'] = xsum/xcont
#     # Ask for arguments for each metric
#
#     return res_dict
#%%
# train_ds,test_ds = read_data(target_type = 1)
#
# # OUTPUTS_a = len(class_names)
#
# list_of_metrics = ['acc', 'hlm']
# list_of_agg = ['avg']
# train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on='acc')