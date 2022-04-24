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
os.chdir('..')
csv_path = os.getcwd()+'/26-29_09_2017_KCL/'
#%%
df = pd.read_csv(csv_path+'KCL_trim_split_spec.csv')
df = df[['label','wav_name']]
df.columns =['label','id']
df['id'] = csv_path+'trimed_audio/'+df['id']
#%%
df1 = df[df['label']=='pd'].sample(200,random_state=42)
df2 = df[df['label']=='hc'].sample(200,random_state=42)
df = pd.concat([df1,df2],ignore_index=True)
#%%
if not os.path.exists(csv_path+'sample_audio'):
    os.makedirs(csv_path+'sample_audio')
if not os.path.exists(csv_path+'sample_image'):
    os.makedirs(csv_path+'sample_image')
sample_rate = 16000
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128
def melspec_librosa(x):
    return librosa.feature.melspectrogram(
    x,
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
      csv_path + '/sample_image/' + file + "_aug"+ ".jpg",
      bbox_inches='tight', pad_inches=0)

  return file + "_aug"+ ".jpg"
file_name_list = []
for path in df['id']:
    file_name_list.append(str(path).split('/')[-1])
#%%
df['File Name'] = file_name_list
def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(y=signal,rate=time_stretch_rate)
def aug_spec(path_column,file_name_column,label_column):
    wav_name_list =[]
    img_name_list=[]
    label_list=[]
    for i in range(len(path_column)):
        audio, sr = librosa.load(path_column[i],sr=16000)
        # out_filename = "split_" + str(counter) + "_" + file_name
        out_filename = csv_path+'/sample_audio/'+'aug_'+file_name_column[i]
        # Write 10 second segment
        wav_name_list.append('aug_'+file_name_column[i])
        # audio= time_stretch(audio,0.5)
        # sf.write(out_filename, audio, sr)
        img_name=save_spectrogram(melspec_librosa(audio),file_name_column[i])
        img_name_list.append(img_name)
        label_list.append(label_column[i])
    return wav_name_list,img_name_list,label_list
#%%
wav,img,label = aug_spec(df['id'],df['File Name'],df['label'])
df1 = pd.DataFrame()
df1['wav_name'] = wav
df1['img_name'] = img
df1['label'] = label
#%%
df1.to_csv(csv_path+'KCL_spec.csv',index=False)