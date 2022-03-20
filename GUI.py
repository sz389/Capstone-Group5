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
import librosa
from torchvision import models
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
from scipy import signal

import gradio as gr
#%%

def spectrogram(audio):
    sr, data = audio
    if len(data.shape) == 2:
        data = np.mean(data, axis=0)
    frequencies, times, spectrogram_data = signal.spectrogram(
        data, sr, window="hamming"
    )
    plt.pcolormesh(times, frequencies, np.log10(spectrogram_data))
    return plt

sample_rate = 16000
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 180
def melspec_librosa(x):
    print(x)
    x,sr = librosa.load(x,sr=16000)
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
def save_spectrogram(spec, aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  im = axs.imshow(librosa.power_to_db(melspec_librosa(spec)), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  axs.get_xaxis().set_visible(False)
  axs.get_yaxis().set_visible(False)
  plt.axis('off')
  plt.tight_layout()
  # fig.canvas.draw()
  # plt.show(block=False)
  plt.savefig(
      csv_path+ '/sample'+ ".jpg",
      bbox_inches='tight', pad_inches=0)
  # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  # data = data.reshape(fig.carnvas.get_width_height()[::-1] + (3,))
  return fig

#%%
os.chdir('..')
csv_path = os.getcwd()+'/26-29_09_2017_KCL/'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#%%
def define_model(category):
    model_path = csv_path +'saved_cnn/'
    if category=='parkinson':
        IMAGE_SIZE = 128
        class_names = ['PD', 'HC']
        model_name = 'efficientnet_b2'
    # model = models.resnet18(pretrained=True)
    # model = models.resnet34(pretrained=True)
    # model = models.vgg16(pretrained=True)
        OUTPUTS_a = len(class_names)
        model = models.efficientnet_b2(pretrained=True)
        # model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,OUTPUTS_a)
        cnn1 = model.to(device)
        cnn1.load_state_dict(torch.load(model_path+"model_{}.pt".format(model_name)))
    else:
        model_name = 'efficientnet_b2'
        # model = models.resnet18(pretrained=True)
        # model = models.resnet34(pretrained=True)
        # model = models.vgg16(pretrained=True)
        OUTPUTS_a = len(class_names)
        # cnn = CNN()
        model = models.efficientnet_b2(pretrained=True)
        # model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, OUTPUTS_a)
        cnn1 = model.to(device)
        cnn1.load_state_dict(torch.load(model_path + "model_{}.pt".format(model_name)))
    return IMAGE_SIZE,cnn1

# file = xdf_dset['id'][150]
# label = xdf_dset['label'][150]
def predict(file,category):
    IMAGE_SIZE, cnn = define_model(category)
    cnn.eval()
    fig = save_spectrogram(file)
    file = csv_path+ '/sample'+ ".jpg"
    img = cv2.imread(file)
    # sigma = 0.155
    # img = random_noise(img,var = sigma**2)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    X = torch.FloatTensor(img)
    X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))
    X = torch.unsqueeze(X,0)
    images = Variable(X).to("cuda")
    outputs = cnn(images).detach().to(torch.device('cpu'))
    #print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    # print(f'True label:{label}')
    return fig,class_names[predicted]
iface = gr.Interface(predict,
                     inputs=[gr.inputs.Audio(source="upload",\
                                                type="filepath", label=None, optional=False),
                             gr.inputs.Radio(["sex", "emotion", "parkinson"]),],
                     outputs=["plot",'text'],
                     # outputs='text',
                     )

iface.test_launch()
iface.launch(share=True)