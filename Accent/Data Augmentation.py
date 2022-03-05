#%%import sys
import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd  # To play sound in the notebook
import librosa
import librosa.display
import tqdm
from sklearn.model_selection import train_test_split

#%%
## ==================== Import Data ====================
import torch.cuda

csv_path = '/home/ubuntu/Capstone/'
df_full = pd.read_csv(csv_path + 'splitmp3.csv', index_col = 0)
df_full = df_full[['filename', 'path', 'native_language']]
# # only use top 10
# top10 = ['english','spanish','arabic','mandarin','french',
#          'korean','portuguese','russian','dutch','turkish']

# only use top 5
top5 = ['english','spanish','arabic','mandarin','french']
df1 = df_full[df_full['native_language'].isin(top5)]
df_full = df1
#%%
df, df_test = train_test_split(df_full, random_state = 42, test_size = 0.2, stratify = df_full['native_language'], shuffle = True)
# df_test.to_csv(csv_path+"Image_Test.csv")
#%%
indexlist  = list(df_test.index)
df19 = pd.read_csv('/home/ubuntu/'+'CapstoneZZ.csv')
use_for_test= df19.iloc[indexlist]
#length = 9999
#use_for_test = use_for_test.sample(329,random_state=42)
#%%
use_for_test.to_csv(csv_path+'Test_Images.csv',index=False)
#%%
#df = df.sample(500,random_state=42)
#df.head()

## ==================== Count Labels ====================
#%%
df_count = df.groupby("native_language").count()[["path"]].reset_index()
df_count
#%%
English_num_samples = df_count['path'][1]
English_num_samples

# take the English out of df_count (only keep the labels that we need to apply augmentation), we need to use df_count for loops later
df_count.drop(df_count.index[[1]], inplace=True)
df_count = df_count.reset_index(drop=True)
df_count
#%%
## ==================== Calculate Augmentation Times For Minority Lables ====================
import math

num = {}
for i in range(len(df_count)):
  times = math.ceil(English_num_samples/df_count['path'][i]) # get the augmentation times for each label
  num[df_count['native_language'][i]] = times - 1
num

## ==================== Function To Generate Specturgram ====================
#%%
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio
augmented_images_path = '/home/ubuntu/Capstone/Augmented_Images/'

sample_rate = 16000
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128
import librosa.feature
def melspec_librosa(x):
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
      augmented_images_path + "Augmented_" + file + "_" + str(
          augmentation_times) + ".jpg",
      bbox_inches='tight', pad_inches=0)

  return file + "split" + str(
      augmentation_times) + ".jpg"


def generate_mel_spectrogram(wav_array, file, augmentation_times):
    samples = wav_array
    sample_rate = 16000
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(samples, sr=sample_rate)
    # Audio(AUDIO_FILE)
    sgram = librosa.stft(samples)
    librosa.display.specshow(sgram)
    # use the mel-scale instead of raw frequency
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    librosa.display.specshow(mel_scale_sgram)
    # use the decibel scale to get the final Mel Spectrogram
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(
        augmented_images_path + "Augmented_" + file + "_" + str(
            augmentation_times) + ".jpg",
        bbox_inches='tight', pad_inches=0)

    return "Augmented_" + file + "_" + str(augmentation_times) + ".jpg"



## ==================== Augmentation Part ====================
#%%
import random
import librosa
import soundfile as sf
import numpy as np
import torchaudio
import torch

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# path
audio_path = '/home/ubuntu/Capstone/Splitmp3/'
#augmented_audio_path = '/home/ubuntu/Capstone/Audio_Whitenoise/'
augmented_images_path = '/home/ubuntu/Capstone/Augmented_MelSpectrogram/'


# Augmentation methods
def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal


def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(y=signal,rate=time_stretch_rate)

def pitch_shift(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=num_semitones)


def random_gain(signal, min_factor=0.1, max_factor=0.12):
    gain_rate = random.uniform(min_factor, max_factor)
    augmented_signal = signal * gain_rate
    return augmented_signal

#%%
# random select augmentation method and factor
from random import shuffle
# funcs = [add_white_noise,time_stretch,pitch_scale, random_gain]
# fun = random.choice(funcs)
# print(fun)
# factor = round(random.uniform(0.4,2), 1)
# print(factor)

def random_method(func1,signal1,sr):
    if func1 == add_white_noise:
        factor = round(random.uniform(0.4, 2), 1)
        return add_white_noise(signal1,factor)

    elif func1 == time_stretch:
        factor = round(random.uniform(0.4, 2), 1)
        return time_stretch(signal1,factor)
    elif func1 == pitch_shift:
        factor = round(random.uniform(1,5), 1)
        return pitch_shift(signal1,sr,factor)
    else:
        return random_gain(signal1,2,4)


#%%
if __name__ == "__main__":
    Augmented_csv = pd.DataFrame()  # add augmentation data into this df, then export to csv at the end
    Augmented_Image_csv = pd.DataFrame()  # add augmentation image into this df, then export to csv at the end
    # n = len(df_count['native_langugage'].unique())

    for label in df_count['native_language']:  # get label name
        #factor = 0.4
        for augmentation_times in range(num[label]):  # get the augmentation times for current label
            # get augmentation method and factor randomly
            funcs = [add_white_noise, time_stretch, pitch_shift, random_gain]
            func1 = random.choice(funcs)

            for i in range(len(df[df['native_language'] == label])):
                individual_audio_file = df[df['native_language'] == label]['filename'].iloc[i]  # get individual audio file name
                signal1, sr = librosa.load(audio_path + individual_audio_file , sr=16000)  # load individual audio file and resample it to 16000
                #augmented_signal = pitch_scale(signal,sr,factor)  # noise_percentage_factor can be changed here for your preference
                augmented_signal = random_method(func1,signal1,sr)
                #save_graph = generate_mel_spectrogram(augmented_signal, individual_audio_file,augmentation_times)  # generate spectrogram for the augmented audio file
                spec = melspec_librosa(augmented_signal)
                save_graph = save_spectrogram(spec, individual_audio_file, augmentation_times)
                # sf.write(augmented_audio_path + 'Augmented_TimeStrech_' + individual_audio_file + "_" + str(
                #     augmentation_times) + '.wav', augmented_signal, sr)

                # Augmented_csv = Augmented_csv.append(
                #     {'Filename': 'Augmented_TimeStrech_' + individual_audio_file + "_" + str(augmentation_times) + ".wav"
                #         , 'Label': label}, ignore_index=True)

                Augmented_Image_csv = Augmented_Image_csv.append(
                    {'Filename': 'Augmented_' + individual_audio_file + "_" + str(augmentation_times) + ".jpg"
                        , 'label': label}, ignore_index=True)
                del signal1,augmented_signal,spec,save_graph
            #factor +=.1
        # for augmentation_times in range(num[label]):
        #     if augmentation_times == 0: # for the second round of augmentation, we use time stretch
        #       for i in range(len(df[df['native_language'] == label])):
        #         individual_audio_file = df[df['native_language'] == label]['filename'].iloc[i] # get individual audio file name
        #         signal, sr = librosa.load(audio_path + individual_audio_file + ".mp3", sr = 16000) # load individual audio file and resample it to 16000
        #         augmented_signal = add_white_noise(signal,0.4) # noise_percentage_factor can be changed here for your preference
        #         spec = melspec_librosa(augmented_signal)
        #         save_graph = save_spectrogram(spec, individual_audio_file, augmentation_times)
        #         #graph = generate_mel_spectrogram(augmented_signal,individual_audio_file, augmentation_times) # generate spectrogram for the augmented audio file
        #         # sf.write(augmented_audio_path +'Augmented_' + individual_audio_file + "_" + str(augmentation_times) +'.wav', augmented_signal,sr)
        #         # Augmented_csv = Augmented_csv.append({'Filename': 'Augmented_' + individual_audio_file + "_" + str(augmentation_times) +".wav"
        #         #                                       , 'Label': label}, ignore_index=True)
        #
        #         Augmented_Image_csv = Augmented_Image_csv.append({'Filename': 'Augmented_' + individual_audio_file + "_" + str(augmentation_times) +".jpg"
        #                                               , 'Label': label}, ignore_index=True)
        #         del signal, augmented_signal, spec, save_graph
        #
        #     elif augmentation_times == 1: # for the second round of augmentation, we use time stretch
        #       for i in range(len(df[df['native_language'] == label])):
        #         individual_audio_file = df[df['native_language'] == label]['filename'].iloc[i] # get individual audio file name
        #         signal, sr = librosa.load(audio_path + individual_audio_file + ".mp3", sr = 16000) # load individual audio file and resample it to 16000
        #         augmented_signal = time_stretch(signal,0.6) # noise_percentage_factor can be changed here for your preference
        #         graph = generate_mel_spectrogram(augmented_signal,individual_audio_file, augmentation_times) # generate spectrogram for the augmented audio file
        #         sf.write(augmented_audio_path +'Augmented_' + individual_audio_file + "_" + str(augmentation_times) +'.wav', augmented_signal,sr)
        #         Augmented_csv = Augmented_csv.append({'Filename': 'Augmented_' + individual_audio_file + "_" + str(augmentation_times) +".wav"
        #                                               , 'Label': label}, ignore_index=True)
        #
        #         Augmented_Image_csv = Augmented_Image_csv.append({'Filename': 'Augmented_' + individual_audio_file + "_" + str(augmentation_times) +".jpg"
        #                                               , 'Label': label}, ignore_index=True)
            # #


        else:
            pass



# Augmented_csv.to_csv(csv_path + 'Augmented_Audio_data_TimeStrech.csv')
Augmented_Image_csv.to_csv(csv_path + 'Augmented_Image_data.csv')
Augmented_Image_csv.groupby(['label']).count()

#%%

