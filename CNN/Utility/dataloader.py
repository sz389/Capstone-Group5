import torch
import numpy as np
import cv2
from torch.utils import data

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
        y=self.df.label.iloc[index]
        labels_ohe = np.zeros(self.OUTPUTS_a)
        for idx, label in enumerate(range(self.OUTPUTS_a)):
            if label == y:
                labels_ohe[idx] = 1
        y = torch.FloatTensor(labels_ohe)
        file_name = self.df.id.iloc[index]
        img = cv2.imread(file_name)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        X = torch.FloatTensor(img)
        X = torch.reshape(X, (3, self.IMAGE_SIZE, self.IMAGE_SIZE))
        return X, y

def dataloader(xdf_dset, OUTPUTS_a, BATCH_SIZE = 64, IMAGE_SIZE=128,shuffle=True):

    params = {'batch_size': BATCH_SIZE,
              'shuffle':shuffle}
    training_set = Dataset(xdf_dset, OUTPUTS_a,IMAGE_SIZE= IMAGE_SIZE, transform=None)
    training_generator = data.DataLoader(training_set, **params)

    return training_generator
