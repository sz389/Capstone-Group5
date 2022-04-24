# This code is data augmentation and it will generate augmented .wav and mel spectrugrams
#%%
import pandas as pd
import librosa.display
import librosa.display
import math
import random
import numpy as np
import argparse
import soundfile as sf
import os
from melspec_librosa import melspec_librosa, save_spectrogram

#%%
# ============================================== define functions ======================================================
# The function below calculate the augmentation times for all minority labels
def preprocess_data(train_set_path):
    #df = pd.read_csv(train_set_path) # import train dataset
    df_count = df.groupby("label").count()[["id"]].reset_index() # count the number of files for each label
    major_times = df_count[df_count['id'] == max(df_count['id'])] # find the number of files of the majority label
    major_label = df_count[df_count['id'] == max(df_count['id'])]['label'] # find the majority label name
    major_label = major_label.to_frame().reset_index()
    df_count = df_count[df_count['label'] != major_label['label'][0]].reset_index(drop=True) # remove the majority label from the dataframe for calculating augmentation times
    num = {}
    for i in range(len(df_count)): # get the augmentation times for each minority label
        times = math.ceil(major_times['id'].iloc[0] / df_count['id'][i])
        num[df_count['label'][i]] = times - 1
    return df_count, num

#%%
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

def time_shift(signal, sr, factor):
    augmented_signal = np.roll(signal,int(sr/factor))
    return augmented_signal


#%% pick augmentation methods randomly
def random_method(func1,signal1,sr):
    if func1 == add_white_noise:
        factor = round(random.uniform(0.4, 2), 2)
        return add_white_noise(signal1,factor)

    elif func1 == time_stretch:
        factor = round(random.uniform(0.1, 1), 2)
        return time_stretch(signal1,factor)
    elif func1 == pitch_shift:
        factor = round(random.uniform(-5,5), 2)
        return pitch_shift(signal1,sr,factor)
    elif func1 == time_shift:
        factor = round(random.uniform(10, 20), 2)
        return time_shift(signal1, sr, factor)
    else:
        return random_gain(signal1,2,4)


#%% call augmentation

def call_augmentation(dataframe, save_path):
    Augmented_audio_csv = pd.DataFrame()  # add augmentation data into this df, then export to csv at the end
    Augmented_Image_csv = pd.DataFrame()  # add augmentation image into this df, then export to csv at the end
    for label in dataframe['label']:  # get label name
        for augmentation_times in range(num[label]):  # get the augmentation times for current label
            funcs = [add_white_noise, time_shift, time_stretch]# pitch_shift,random_gain]
            func1 = random.choice(funcs)

            for i in range(len(df[df['label'] == label])):
                individual_audio_file = df[df['label'] == label]['id'].iloc[i]  # get individual audio file name
                individual_audio_file_name = individual_audio_file.split('/')[-1]
                signal1, sr = librosa.load(individual_audio_file , sr=16000)  # load individual audio file and resample it to 16000
                augmented_signal = random_method(func1,signal1,sr)
                spec = melspec_librosa(augmented_signal)
                save_graph = save_spectrogram(spec, individual_audio_file_name + '_' + str(augmentation_times), save_path + '/augmented_images/')
                sf.write(save_path + 'augmented_audio/' + 'Augmented_' +  individual_audio_file_name + "_" + str(
                    augmentation_times) + '.wav', augmented_signal, sr)

                Augmented_audio_csv = Augmented_audio_csv.append(
                    {'Filename': 'Augmented_' + individual_audio_file_name + "_" + str(augmentation_times) + ".wav"
                        , 'Label': label}, ignore_index=True)

                Augmented_Image_csv = Augmented_Image_csv.append(
                    {'Filename': 'Augmented_' + individual_audio_file_name + "_" + str(augmentation_times) + ".jpg"
                        , 'label': label}, ignore_index=True)
                del signal1,augmented_signal,spec,save_graph
        else:
            pass

    Augmented_Image_csv.to_csv(save_path + '/augmented_csv/' + 'Augmented_Images.csv')
    Augmented_audio_csv.to_csv(save_path + '/augmented_csv/' + 'Augmented_Audio.csv')
    return Augmented_Image_csv, Augmented_audio_csv

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder", default=None, type=str, required=True)  # Path of csv to load
    parser.add_argument("--save_path", default=None, type=str, required=True)  # Path to save the csv file
    args = parser.parse_args()
    folder = args.csv_folder
    save_folder = args.save_path

    if not os.path.exists(save_folder + '/augmented_csv/'):
        os.makedirs(save_folder + '/augmented_csv/')
    if not os.path.exists(save_folder + '/augmented_audio/'):
        os.makedirs(save_folder + '/augmented_audio/')
    if not os.path.exists(save_folder + '/augmented_images/'):
        os.makedirs(save_folder + '/augmented_images/')

#%%
    df = pd.read_csv(folder + '/data_csv_0326accent_train_trim_split.csv')
    df['id'] = folder + 'trimed_audio/' + df['wav_name']
#%%
    train_set_path = df
    df_count, num = preprocess_data(train_set_path)

    Augmented_Image_csv, Augmented_audio_csv = call_augmentation(df_count, save_folder)


