# %% --------------------------------------- Reading in CSV File with Labels -------------------------------------------------------------------
import pandas as pd
audio_path = 'AudioWAV/'
df = pd.read_csv('CREMA_D_nopath.csv', index_col = 0)
df = df[['File Name', 'emotion']]

df['audio_path'] = audio_path + df['File Name']
print(df.head())


# %% --------------------------------------- Filter broken and non-existed paths -------------------------------------------------------------------
print(f"Step 0: {len(df)}")
import os

df["status"] = df["audio_path"].apply(lambda path: True if os.path.exists(path) else None)
df = df.dropna(subset=["audio_path"])
df = df.drop("status", 1)
print(f"Step 1: {len(df)}")

df = df.sample(frac=1)
df = df.reset_index(drop=True)
df.head()

# %% ------------------------------------- Print our unique labels -----------------------------------------------------
print("Labels: ", df["emotion"].unique())
print()
df.groupby("emotion").count()[["audio_path"]]

import matplotlib.pyplot as plt
import torch

# Plot Spectrograms
def plot_specgram(waveform, sample_rate, xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    # figure.suptitle(title)
    # plt.show(block=False)
    return figure


import torchaudio

def extract_waveform_and_sample_rate(df):
    path_list = []
    i = 0
    for path in df['audio_path']:
        waveform, sample_rate = torchaudio.load(path)
        figure = plot_specgram(waveform, sample_rate)
        file_name = df["File Name"][i]
        # Change savefig File Path to where you want images to be saved
        plt.savefig(f"Spectrogram_Images/{file_name}.jpg",
                    bbox_inches='tight', pad_inches=0)
        # Change path_list file path to where images were saved
        path_list.append(
            f"Spectrogram_Images/{file_name}.jpg")
        i = i + 1
    return path_list

#Change size of df to full only if GPU is on
path_list = extract_waveform_and_sample_rate(df)

df['image_name'] = df['File Name'] + ".jpg"

print(df['image_name'])

print()

print(df.head())

df.to_csv("CREMA_D_Image.csv") #This CSV File will be used by CNN code file



