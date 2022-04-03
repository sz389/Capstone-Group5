import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
import librosa

labels = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']


class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, df, feature_extractor, transform=None):
        # Initialization
        self.df = df
        self.feature_extractor = feature_extractor
        self.transform = transform
    def __len__(self):
        # Denotes the total number of samples'
        return len(self.df)
    def __getitem__(self, index):
        # Generates one sample of data'
        # Select sample
        # Load data and get label
        y=self.df.label.iloc[index]
        file_name = self.df.id.iloc[index]
        X,sr = librosa.load(file_name,sr=self.feature_extractor.sampling_rate)
        dict = {'input_values':X,'label':y}
        return dict


def dataloader(df, feature_extractor):

    data_loader = Dataset(df, feature_extractor)
    return data_loader