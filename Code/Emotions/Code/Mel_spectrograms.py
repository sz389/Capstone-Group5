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
csv_path = '/home/ubuntu/capstone/'
#%%

df = pd.read_csv(csv_path+'CREMA_D_nopath.csv')
df = df[['Race','emotion','File Name']]
df['wav_path'] = csv_path + 'AudioWAV/'+df['File Name']

#%%
if not os.path.exists(csv_path+'Mel_Spectrograms'):
    os.makedirs(csv_path+'Mel_Spectrograms')

#Best for emotion:
sample_rate = 16000
n_fft = 1536
win_length = None
hop_length = 512
n_mels = 160

#Best for race:
# sample_rate = 16000
# n_fft = 1024
# win_length = None
# hop_length = 512
# n_mels = 160

def melspec_librosa(x):
    return librosa.feature.melspectrogram(
    y=x,
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
def save_spectrogram(spec,file, aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  axs.get_xaxis().set_visible(False)
  axs.get_yaxis().set_visible(False)
  plt.axis('off')
  # plt.show(block=False)
  plt.savefig(
      csv_path + 'Mel_Spectrograms/' + file + ".jpg",
      bbox_inches='tight', pad_inches=0)

  return file + ".jpg"

#%%
def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(y=signal,rate=time_stretch_rate)
def aug_spec(path_column,file_name_column,label_column1, label_column2):
    #wav_name_list =[]
    img_name_list=[]
    label_list1=[]
    label_list2 = []
    for i in range(len(path_column)):
        audio, sr = librosa.load(path_column[i],sr=16000)
        # out_filename = "split_" + str(counter) + "_" + file_name
        # out_filename = csv_path+'/sample_audio/'+'aug_'+file_name_column[i]
        # Write 10 second segment
        #wav_name_list.append('aug_'+file_name_column[i])
        # audio= time_stretch(audio,0.5)
        # sf.write(out_filename, audio, sr)
        img_name=save_spectrogram(melspec_librosa(audio),file_name_column[i]+'_'+str(n_fft)+'_'+str(hop_length)+"_"+str(n_mels))
        img_name_list.append(img_name)
        label_list1.append(label_column1[i])
        label_list2.append(label_column2[i])
    return img_name_list,label_list1,label_list2
#%%
img,label1,label2 = aug_spec(df['wav_path'],df['File Name'],df['Race'],df['emotion'])
df1 = pd.DataFrame()
df1['img_name'] = img
df1['Race'] = label1
df1['emotion'] = label2
#%%
df1.to_csv(csv_path+'Mel_Spectrograms_1536_512_160.csv',index=False)
