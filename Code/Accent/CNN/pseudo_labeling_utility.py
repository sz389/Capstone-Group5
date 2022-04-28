from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def manual_label_encoder(df_label, category = "emotion"):
    label_list = []
    if category == "parkinson":
        for label in df_label:
            if label == 'hc':
                label_list.append(0)
            else:
                label_list.append(1)

    elif category == "age":
        for label in df_label:
            if label == "<30s":
                label_list.append(0)
            elif label == "30s":
                label_list.append(1)
            elif label == "40s":
                label_list.append(2)
            elif label == "50s":
                label_list.append(3)
            elif label == ">60s":
                label_list.append(4)

    elif category == "race":
        for label in df_label:
            if label == "Caucasian":
                label_list.append(0)
            elif label == "African American":
                label_list.append(1)
            elif label == "Asian":
                label_list.append(2)

    elif category == 'sex':
        for label in df_label:
            if label == "Male":
                label_list.append(0)
            else:
                label_list.append(1)

    elif category == "emotion":
        for label in df_label:
            if label == "ANG":
                label_list.append(0)
            elif label == 'DIS':
                label_list.append(1)
            elif label == "FEA":
                label_list.append(2)
            elif label == "HAP":
                label_list.append(3)
            elif label == "NEU":
                label_list.append(4)
            elif label == "SAD":
                label_list.append(5)

    elif category == "accent":
        for label in df_label:
            if label == "arabic":
                label_list.append(0)
            elif label == "english":
                label_list.append(1)
            elif label == "french":
                label_list.append(2)
            elif label == "mandarin":
                label_list.append(3)
            else:
                label_list.append(4)

    return label_list
#%%

import torch
import numpy as np
import cv2
from torch.utils import data
#%%
def get_filename(path_list):
    file_name_list = []
    for path in path_list:
        file_name_list.append(str(path).split('/')[-1])
    return file_name_list
def hard_code_parkinson(df1_label):
    label_list = []
    for i in df1_label:
        if i =='hc':
            label_list.append(0)
        else:
            label_list.append(1)
    return label_list
#%%
class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''

    def __init__(self, df, OUTPUTS_a, transform=None, IMAGE_SIZE=128):
        self.df = df
        self.transform = transform
        self.IMAGE_SIZE = IMAGE_SIZE
        self.OUTPUTS_a = OUTPUTS_a

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        y=self.df.id.iloc[index]
        labels_ohe = np.zeros(self.OUTPUTS_a)
        for idx, label in enumerate(range(self.OUTPUTS_a)):
            if label == y:
                labels_ohe[idx] = 1
        y = torch.FloatTensor(labels_ohe)
        file_name = self.df.id.iloc[index]
        img = cv2.imread(file_name)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        X = torch.FloatTensor(img)
        X = torch.reshape(X, (3, self.IMAGE_SIZE, self.IMAGE_SIZE))/255
        return X, y

def dataloader(xdf_dset, OUTPUTS_a, BATCH_SIZE = 64, IMAGE_SIZE=128,shuffle=True):

    params = {'batch_size': BATCH_SIZE,
              'shuffle':shuffle}
    training_set = Dataset(xdf_dset, OUTPUTS_a,IMAGE_SIZE= IMAGE_SIZE, transform=None)
    training_generator = data.DataLoader(training_set, **params)

    return training_generator
#%%
def get_n_params(model):
    """
    Reference: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    :param model:
    :return:
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class UnlabeledDataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''

    def __init__(self, df, transform=None, IMAGE_SIZE = 128):
        self.df = df
        self.transform = transform
        self.IMAGE_SIZE = IMAGE_SIZE
        #self.OUTPUTS_a = OUTPUTS_a

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_name = self.df.id.iloc[index]
        img = cv2.imread(file_name)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        X = torch.FloatTensor(img)
        X = torch.reshape(X, (3, self.IMAGE_SIZE, self.IMAGE_SIZE))/255
        return X




def unlabeled_dataloader(xdf_dset,BATCH_SIZE = 64, IMAGE_SIZE=128,shuffle=True):

    params = {'batch_size': BATCH_SIZE,
              'shuffle':shuffle}
    training_set = UnlabeledDataset(xdf_dset,IMAGE_SIZE= IMAGE_SIZE, transform=None)
    training_generator = data.DataLoader(training_set, **params)

    return training_generator


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    eval_labels = eval_pred.label_ids
    predictions = np.argmax(eval_pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(eval_labels, predictions, average='micro')
    acc = accuracy_score(eval_labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_model_params(model):
    for param in model.parameters():
        print(param.data)


def get_filename(path_list):
    file_name_list = []
    for path in path_list:
        file_name_list.append(str(path).split('/')[-1])
    return file_name_list

def get_classes(category):
    if category == "race":
        return ['Caucasian','African American', 'Asian']
    elif category == "age":
        return [ '<30s', '30s', '40s', '50s','>60s']
    elif category == 'sex':
        return ['Male','Female']
    elif category == "emotion":
        return ["ANG",'DIS','FEA','HAP','NEU','SAD']
    elif category =='accent':
        return ['arabic','english', 'french', 'mandarin', 'spanish']
