#%%
# This file is used to import data, add path column in the dataframe, and split data to train, validation, and test sets.
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def import_data(csv_path, audio_path, csv_name):
    # read ORIGINAL audio files from csv
    columns = ['filename', 'native_language']
    df = pd.read_csv(csv_path + csv_name, usecols=columns)

    # only use 5 labels
    class_names = ['arabic', 'english', 'french', 'mandarin', 'spanish']
    # filter df based on 5 labels
    df_5 = df[df['native_language'].isin(class_names)]
    # add audio path in the dataframe
    original_audio_path = audio_path
    df_5['path'] = original_audio_path + df_5['filename'] + '.mp3'

    df_train, df_test = train_test_split(df_5, random_state=42, test_size=0.2, stratify=df_5['native_language'])
    df_train, df_val = train_test_split(df_train, random_state=42, test_size=0.1, stratify=df_train['native_language'])

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_val.reset_index(inplace=True, drop=True)

    df_train.to_csv(csv_path + 'accent_train.csv', index=False)
    df_test.to_csv(csv_path  + 'accent_test.csv', index=False)
    df_val.to_csv(csv_path   + 'accent_val.csv', index=False)

    return df_train, df_test, df_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder", default=None, type=str, required=True)  # Path of csv to load
    parser.add_argument("--csv_name",   default=None, type=str, required=True)
    parser.add_argument("--audio_folder", default=None, type=str, required=True)  # Path to save the csv file
    args = parser.parse_args()
    csv_path = args.csv_folder
    audio_path = args.audio_folder
    csv_name = args.csv_name

    import_data(csv_path,audio_path, csv_name)



