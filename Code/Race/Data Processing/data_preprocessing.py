'''
creating csv file from original csv file
label, Audio_file_name, Audio_file_path, Image_file_name, Image_file_path
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def train_val_test_split(df, category, csv_save_path): #split into 72% train, 8% validation, 20% test
    #we get from generate_mel_spectrogram.py
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[category])
    train, val = train_test_split(train, test_size=0.1, random_state=42,
                                                  stratify=train[category])

    train.to_csv(csv_save_path + f"{category}_train.csv", index=False)
    val.to_csv(csv_save_path + f"{category}_val.csv", index=False)
    test.to_csv(csv_save_path + f"{category}_test.csv", index=False)


def generate_csv(csv_file, csv_path, audio_path, category):
    df1 = pd.read_csv(csv_path + csv_file)
    df1 = df1[df1["race"] != "Unknown"]
    df1['Audio_file_name'] = df1['File Name']
    df1['Audio_file_path'] = audio_path + df1['Audio_file_name']
    df = df1[[category, "Audio_file_name", "Audio_file_path"]]

    df.to_csv(csv_path + f"{category}_with_audio_path.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="CREMA_no_path.csv", type=str, required=True)  # CREMA_no_path.csv
    parser.add_argument("--csv_path", default=None, type=str, required=True)  # Path to save the csv file
    parser.add_argument('--audio_path',default=None,type =str, required=True) # path of AudioWAV folder
    parser.add_argument('--category', default=None, type=str, required=True) # category (Ex. emotion, race, etc.)
    args = parser.parse_args()
    csv_file = args.csv_file
    csv_path = args.csv_path
    audio_path = args.audio_path
    category = args.category

    generate_csv(csv_file, csv_path, audio_path, category)
    #will generate csv file with 3 columns: label, audio file name and audio file path





