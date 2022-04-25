#%%
import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd  # To play sound in the notebook
import librosa
import librosa.display
import tqdm
import soundfile as sf
import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd  # To play sound in the notebook
import librosa
import librosa.display
import tqdm
import soundfile as sf
import argparse
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
        y=self.df.label_num.iloc[index]
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
