#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
# from CNN.utility import get_filename
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

def generate_spec(path_column,file_name_column,label_column,save_path):
    img_name_list=[]
    label_list=[]
    for i in range(len(path_column)):
        audio, sr = librosa.load(path_column[i],sr=16000)
        img_name=save_spectrogram(melspec_librosa(audio,n_mels=180),file_name_column[i],save_path+'/mel_spectrograms/')
        img_name_list.append(img_name)
        label_list.append(label_column[i])
    return img_name_list,label_list

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv_path", default=None, type=str, required=True)  # Path of csv to load
#     parser.add_argument("--save_path", default=None, type=str, required=True)  # Path to save the csv file
#     parser.add_argument('--csv_csv',default=None,type = str, required=True) # csv file name to save
#     args = parser.parse_args()
#     csv_path = args.csv_path
#     save_path = args.save_path
#     csv_name = args.save_csv
#     if not os.path.exists(save_path+'/mel_spectrograms'):
#         os.makedirs(save_path+'/mel_spectrograms')
#     df = pd.read_csv(csv_path)
#     filename = get_filename(df['id'])
#     img,label = generate_spec(df['id'],filename,df['label'],save_path)
#     df12 = pd.DataFrame()
#     df12['id'] = img
#     df12['label'] = label
#     df12.to_csv(save_path+'/'+csv_name,index=False)


