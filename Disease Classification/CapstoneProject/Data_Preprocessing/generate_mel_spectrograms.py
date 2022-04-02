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
#%%
def melspec_librosa(x,sample_rate = 16000,
    n_fft = 1024,
    win_length = None,
    hop_length = 512,
    n_mels = 180):
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
def save_spectrogram(spec,file,path, aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  axs.get_xaxis().set_visible(False)
  axs.get_yaxis().set_visible(False)
  plt.axis('off')
  # plt.show(block=False)
  plt.savefig(
      path+'/' + file + ".jpg",
      bbox_inches='tight', pad_inches=0)
  return path+'/'+file + ".jpg"
def generate_spec(path_column,file_name_column,label_column,origin,label_num):
    wav_name_list =[]
    origin_list = []
    img_name_list=[]
    label_list=[]
    labelnum_list = []
    for i in range(len(path_column)):
        audio, sr = librosa.load(path_column[i],sr=16000)
        # out_filename = "split_" + str(counter) + "_" + file_name
        # out_filename = csv_path+'/sample_audio/'+'aug_'+file_name_column[i]
        # Write 10 second segment
        # wav_name_list.append('aug_'+file_name_column[i])
        # audio= time_stretch(audio,0.5)
        # sf.write(out_filename, audio, sr)
        img_name=save_spectrogram(melspec_librosa(audio),file_name_column[i])
        img_name_list.append(img_name)
        origin_list.append(origin[i])
        labelnum_list.append(label_num[i])
        label_list.append(label_column[i])
    return img_name_list,label_list,labelnum_list,origin_list