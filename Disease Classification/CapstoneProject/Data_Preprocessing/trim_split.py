#%%
import os
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
import argparse
from generate_mel_spectrograms import melspec_librosa,save_spectrogram
#%%
def trim_split(path_column,file_name_column,label_column,csv_path):
# https://stackoverflow.com/questions/60105626/split-audio-on-timestamps-librosa
# First load the file
    wav_name_list =[]
    label_list=[]
    img_name_list = []
    origin_list = []
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
            img_name = save_spectrogram(melspec_librosa(audio), 'split_'+str(counter)+'_'+file_name_column[i],
                                        save_path+'/mel_spectrograms/')
            img_name_list.append(img_name)
            # out_filename = "split_" + str(counter) + "_" + file_name
            out_filename = csv_path+'/trimed_audio/'+'split_'+str(counter)+'_'+file_name_column[i]
            # Write 10 second segment
            wav_name_list.append(out_filename)
            origin_list.append(file_name_column[i])
            sf.write(out_filename, block, sr)
            label_list.append(label_column[i])
            counter += 1
            samples_wrote += buffer
    return wav_name_list,img_name_list,label_list,origin_list
def get_filename(path_list):
    file_name_list = []
    for path in path_list:
        file_name_list.append(str(path).split('/')[-1])
    return file_name_list
def hard_code_parkinson(df1_label):
    label_list = []
    for i in df1_label:
        if i =='hc':
            label_list.append(0)
        else:
            label_list.append(1)
    return label_list
def save_df_csv(wav,img,label,origin):
    df12 = pd.DataFrame()
    df12['id'] = wav
    df12['label'] = label
    df12['origin'] = origin
    df12['label_num'] = hard_code_parkinson(df12['label'])

    dfimg = pd.DataFrame()
    dfimg['id'] = img
    dfimg['label'] = label
    dfimg['origin'] = origin
    dfimg['label_num'] = hard_code_parkinson(dfimg['label'])
    return df12, dfimg

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--csv_folder", default=None, type=str, required=True)  # Path of csv to load
    # parser.add_argument("--save_path", default=None, type=str, required=True)  # Path to save the csv file
    # args = parser.parse_args()
    # csv_path = args.csv_folder
    # save_path = args.save_path
    csv_path = "/home/ubuntu/Capstone/data/"
    save_path = "/home/ubuntu/Capstone/data/"
    if not os.path.exists(save_path + '/trimed_audio'):
        os.makedirs(csv_path + '/trimed_audio')
    if not os.path.exists(save_path + '/mel_spectrograms'):
        os.makedirs(csv_path + '/mel_spectrograms')
    df = pd.read_csv(csv_path+'/Parkinsontrain.csv')
    filename = get_filename(df['id'])
    wav,img,label, origin = trim_split(df['id'], filename, df['label'],save_path)
    df12,dfimg = save_df_csv(wav,img,label,origin)
    df12.to_csv(save_path+'/KCL_train_trim_split_audio.csv',index=False)
    dfimg.to_csv(save_path +'/KCL_train_trim_split_spec.csv',index=False)
    df1 = pd.read_csv(csv_path+'Parkinsontest.csv')
    filename = get_filename(df1['id'])
    wav,img,label, origin = trim_split(df1['id'], filename, df1['label'],save_path)
    df12,dfimg =save_df_csv(wav,img,label,origin)
    df12.to_csv(save_path+'/KCL_test_trim_split_audio.csv',index=False)
    dfimg.to_csv(save_path +'/KCL_test_trim_split_spec.csv',index=False)



