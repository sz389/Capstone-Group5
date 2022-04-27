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
import argparse
import sys
sys.path.insert(1, '/home/ubuntu/capstone/Data_Processing')
from data_preprocessing import train_val_test_split

#%%
def melspec_librosa(x,sample_rate = 16000,
    n_fft = 1024,
    win_length = None,
    hop_length = 512,
    n_mels = 160):
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
def generate_spec(path_column,file_name_column,save_path):
    # wav_name_list =[]
    # origin_list = []
    img_name_list=[]
    # labelnum_list = []
    for i in range(len(path_column)):
        audio, sr = librosa.load(path_column[i],sr=16000)
        # out_filename = "split_" + str(counter) + "_" + file_name
        # out_filename = csv_path+'/sample_audio/'+'aug_'+file_name_column[i]
        # Write 10 second segment
        # wav_name_list.append('aug_'+file_name_column[i])
        # audio= time_stretch(audio,0.5)
        # sf.write(out_filename, audio, sr)
        img_name=save_spectrogram(melspec_librosa(audio,n_mels=180),file_name_column[i],save_path+'/mel_spectrograms/')
        img_name_list.append(img_name)

        # origin_list.append(origin[i])
        # labelnum_list.append(label_num[i])
    return img_name_list

def save_spectrogram_gui(spec,file,path, aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  axs.get_xaxis().set_visible(False)
  axs.get_yaxis().set_visible(False)
  plt.axis('off')
  plt.tight_layout()
  # plt.show(block=False)
  plt.savefig(
      path+'/' + file + ".jpg",
      bbox_inches='tight', pad_inches=0)
  return path+'/'+file + ".jpg",fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_csv_path", default=None, type=str, required=True)  # Path of csv to load
    parser.add_argument("--save_spec_path", default=None, type=str, required=True)  # path to store spectrograms
    parser.add_argument("--save_csv_path", default=None, type=str, required=True) #path to save csv
    parser.add_argument("--category", default=None, type=str, required=True)  # #category (Ex. emotion, race, etc.)
    args = parser.parse_args()
    load_csv_path = args.load_csv_path
    save_spec_path = args.save_spec_path
    save_csv_path = args.save_csv_path
    category = args.category
    if not os.path.exists(save_spec_path+'/mel_spectrograms'):
        os.makedirs(save_spec_path+'/mel_spectrograms')

    df = pd.read_csv(load_csv_path + f"{category}_with_audio_path.csv")
    img = generate_spec(df['Audio_file_path'],df['Audio_file_name'], save_spec_path) #img = list of file paths
    df['Image_file_path'] = img

    train_val_test_split(df, category, save_csv_path) #saves train, validation and test csv file
