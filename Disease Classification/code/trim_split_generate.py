#%%
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import torch
import pandas as pd
import os
import torchaudio
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa.display
#%%
os.chdir('..')
csv_path = os.getcwd()+'/26-29_09_2017_KCL/'
if not os.path.exists(csv_path+'trimed_audio'):
    os.makedirs(csv_path+'trimed_audio')
if not os.path.exists(csv_path+'mel_spectrograms'):
    os.makedirs(csv_path+'mel_spectrograms')
#%%
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
def save_spectrogram(spec,file,augmentation_times, aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  axs.get_xaxis().set_visible(False)
  axs.get_yaxis().set_visible(False)
  plt.axis('off')
  # plt.show(block=False)
  plt.savefig(
      csv_path + '/mel_spectrograms/' + file + "_split_" + str(
          augmentation_times) + ".jpg",
      bbox_inches='tight', pad_inches=0)

  return file + "_split_" + str(
      augmentation_times) + ".jpg"
#%%
df = pd.read_csv(csv_path+'Parkinson.csv')
file_name_list = []
for path in df['path']:
    file_name_list.append(str(path).split('/')[-1])
#%%
df['File Name'] = file_name_list
#%%
def trim_split_generate(path_column,file_name_column,label_column):
# https://stackoverflow.com/questions/60105626/split-audio-on-timestamps-librosa
# First load the file
    wav_name_list =[]
    img_name_list=[]
    label_list=[]
    for i in range(len(path_column)):
        audio, sr = librosa.load(path_column[i],sr=16000)
        # split by silence and combine together
        wav_data = []
        clips = librosa.effects.split(audio, top_db=32)
        for c in clips:
            data = audio[c[0]: c[1]]
            wav_data.extend(data)
        # Get number of samples for 2 seconds; replace 2 by any number
        buffer = 10 * sr

        samples_total = len(wav_data)
        samples_wrote = 0
        counter = 1

        while samples_wrote < samples_total:

            #check if the buffer is not exceeding total samples
            if buffer > (samples_total - samples_wrote):
                buffer = samples_total - samples_wrote

            block = wav_data[samples_wrote : (samples_wrote + buffer)]
            block = np.array(block)
            # out_filename = "split_" + str(counter) + "_" + file_name
            out_filename = csv_path+'/trimed_audio/'+'split_'+str(counter)+'_'+file_name_column[i]
            # Write 10 second segment
            wav_name_list.append('split_'+str(counter)+'_'+file_name_column[i])
            sf.write(out_filename, block, sr)
            img_name=save_spectrogram(melspec_librosa(block),file_name_column[i],counter)
            img_name_list.append(img_name)
            label_list.append(label_column[i])
            counter += 1
            samples_wrote += buffer
    return wav_name_list,img_name_list,label_list
wav,img,label = trim_split_generate(df['path'],df['File Name'],df['Disease'])
df1 = pd.DataFrame()
df1['wav_name'] = wav
df1['img_name'] = img
df1['label'] = label
#%%
df1.to_csv(csv_path+'KCL_trim_split_spec.csv',index=False)