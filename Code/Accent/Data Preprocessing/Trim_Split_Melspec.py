#%%
import numpy as np
import pandas as pd
import os
import librosa
import soundfile as sf
import librosa.display
from Finalized_Code.Data_Preprocessing.melspec_librosa import melspec_librosa ,save_spectrogram
import argparse
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
        buffer = 5 * sr
        samples_total = len(audio)
        samples_wrote = 0
        counter = 1
        while samples_wrote < samples_total:
            #check if the buffer is not exceeding total samples
            if buffer > (samples_total - samples_wrote):
                buffer = samples_total - samples_wrote
            block = audio[samples_wrote : (samples_wrote + buffer)]
            block = np.array(block)
            img_name = save_spectrogram(melspec_librosa(block), 'split_'+str(counter)+'_'+file_name_column[i],
                                        save_path+'/new_mel_spectrograms/')
            img_name_list.append(img_name)
            # out_filename = "split_" + str(counter) + "_" + file_name
            out_filename = save_path+'/new_trimed_audio/'+'split_'+str(counter)+'_'+file_name_column[i] + '.wav'
            # Write 10 second segment
            wav_name_list.append(out_filename)
            origin_list.append(file_name_column[i])
            sf.write(out_filename, block, sr)
            label_list.append(label_column[i])
            counter += 1
            samples_wrote += buffer
    return wav_name_list,img_name_list,label_list,origin_list

def hard_code_accent(row):
    if row == 'arabic':
        return 0
    elif row== 'english':
        return 1
    elif row == 'french':
        return 2
    elif row== 'mandarin':
        return 3
    else: return 4

def save_df_csv(wav,img,label,origin):
    df12 = pd.DataFrame()
    df12['path'] = wav
    df12['native_language'] = label
    df12['origin'] = origin
    list_num = []
    for i in df12['native_language']:
        list_num.append(hard_code_accent(i))
    df12['label_num'] = list_num

    dfimg = pd.DataFrame()
    dfimg['path'] = img
    dfimg['native_language'] = label
    dfimg['origin'] = origin
    list_num = []
    for i in df12['native_language']:
        list_num.append(hard_code_accent(i))
    dfimg['label_num'] = list_num
    return df12, dfimg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder", default=None, type=str, required=True)  # Path of csv to load
    parser.add_argument("--save_path", default=None, type=str, required=True)  # Path to save the csv file
    args = parser.parse_args()
    csv_path = args.csv_folder
    save_path = args.save_path

    # csv_path = "/home/ubuntu/Capstone/data_train_val_test/"
    # save_path = "/home/ubuntu/Capstone/data_train_val_test/"
    if not os.path.exists(save_path + '/new_trimed_audio'):
        os.makedirs(save_path + '/new_trimed_audio')
    if not os.path.exists(save_path + '/new_mel_spectrograms'):
        os.makedirs(save_path + '/new_mel_spectrograms')

#%%
    df_train = pd.read_csv(csv_path + 'accent_train.csv')
    df_test = pd.read_csv(csv_path + 'accent_test.csv')
    df_val = pd.read_csv(csv_path + 'accent_val.csv')
    #%%
    wav,img,label, origin = trim_split(df_train['path'], df_train['filename'], df_train['native_language'],save_path)
    df12,dfimg = save_df_csv(wav,img,label,origin)
    df12.to_csv(csv_path+'/new_train_trim_split_audio.csv',index=False)
    dfimg.to_csv(csv_path +'/new_train_trim_split_spec.csv',index=False)

    wav,img,label, origin = trim_split(df_val['path'], df_val['filename'], df_val['native_language'],save_path)
    df12,dfimg =save_df_csv(wav,img,label,origin)
    df12.to_csv(csv_path+'/new_val_trim_split_audio.csv',index=False)
    dfimg.to_csv(csv_path +'/new_val_trim_split_spec.csv',index=False)

    wav,img,label, origin = trim_split(df_test['path'], df_test['filename'], df_test['native_language'],save_path)
    df12,dfimg =save_df_csv(wav,img,label,origin)
    df12.to_csv(csv_path+'/new_test_trim_split_audio.csv',index=False)
    dfimg.to_csv(csv_path +'/new_test_trim_split_spec.csv',index=False)

    print('finally the files are generated!!!')

