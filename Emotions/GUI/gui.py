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
import sys
sys.path.insert(1, '/home/ubuntu/capstone/')
from Data_Processing.generate_mel_spectrograms import save_spectrogram_gui, melspec_librosa
import gradio as gr
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, pretrained_model, CNN9


#%%
def get_arg(category):
    #class_names = []; IMAGE_SIZE = 128; model = None; csv_path = ""; model_path = ""
    if category == 'parkinson':
        class_names = ['hc','pd']
        IMAGE_SIZE = 128
        model = pretrained_model('resnet34',2)
        csv_path = "/home/ubuntu/capstone/GUI/upload_images/"
        model_path = '/home/ubuntu/capstone/CNN/Models/Saved_Models/resnet34_parkinson.pt'

    elif category == 'emotion':
        class_names = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
        IMAGE_SIZE = 128
        model = CNN9(len(class_names))
        csv_path = "/home/ubuntu/capstone/GUI/upload_images/"
        model_path = '/home/ubuntu/capstone/CNN/Models/Saved_Models/CNN9_emotion.pt'

    elif category == 'age':
        class_names =  [ '<30s', '30s', '40s', '50s','>60s']
        IMAGE_SIZE = 128
        model = pretrained_model('resnet18', len(class_names))
        csv_path = "/home/ubuntu/capstone/GUI/upload_images/"
        model_path = '/home/ubuntu/capstone/CNN/Models/Saved_Models/cnn_resnet18_age.pt'

    elif category == 'sex':
        class_names = ['male', 'female']
        IMAGE_SIZE = 128
        model = pretrained_model('resnet18', len(class_names))
        csv_path = "/home/ubuntu/capstone/GUI/upload_images/"
        model_path = '/home/ubuntu/capstone/CNN/Models/Saved_Models/cnn_resnet18_sex.pt'

    elif category == 'accent':
        class_names = ['arabic', 'english', 'french', 'mandarin' ,'spanish']
        IMAGE_SIZE = 128
        model = pretrained_model('resnet34', len(class_names))
        csv_path = "/home/ubuntu/capstone/GUI/upload_images/"
        model_path = '/home/ubuntu/capstone/CNN/Models/Saved_Models/resnet34_accent.pt'

    return class_names, IMAGE_SIZE, model, csv_path, model_path


def predict(file,category):
    x, sr = librosa.load(file, sr=16000)
    # IMAGE_SIZE, cnn = define_model(category)
    class_names, IMAGE_SIZE, model, csv_path,model_path = get_arg(category)

    model.load_state_dict(torch.load(f=model_path))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cnn = model.to(device)

    cnn.eval()

    spec = melspec_librosa(x,sample_rate = 16000,
    n_fft = 1536,
    win_length = None,
    hop_length = 512,
    n_mels = 160)
    fig,pic = save_spectrogram_gui(spec,'test',csv_path)

    img = cv2.imread(fig)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    X = torch.FloatTensor(img)
    X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))/255
    X = torch.unsqueeze(X,0)

    images = Variable(X).to("cuda")
    outputs = cnn(images).detach().to(torch.device('cpu'))
    #print(outputs)

    import torch.nn.functional as nnf
    prob = nnf.softmax(outputs, dim=1)
    top_p, top_class = prob.topk(len(class_names), dim=1)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    # print(f'True label:{label}')=

    text = ''
    for i in range(len(class_names)):
        temp = f'{class_names[top_class[0][i]]}: {top_p[0][i]*100:.2f}%\n'
        text  = text+temp
    return pic,text
    # return pic,class_names[predicted]

if __name__=='__main__':
    iface = gr.Interface(predict,
                         inputs=[gr.inputs.Audio(source="microphone", \
                                                 type="filepath", label=None, optional=False),
                                 gr.inputs.Radio(["sex","age" ,"emotion", 'accent', "parkinson"]), ],
                         outputs=["plot", 'text'],
                         # outputs='text',
                         )

    #iface.test_launch()
    iface.launch(share=True)
    # predict("/home/ubuntu/Capstone/data/trimed_audio/split_9_ID36_hc_0_0_0.wav",'parkinson')