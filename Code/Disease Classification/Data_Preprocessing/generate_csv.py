#%%
import os
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
import os
import argparse
import torchaudio
from sklearn.model_selection import train_test_split
#%%
def generate_csv_from_path(data_path):
    data = []
    for path in tqdm(Path(data_path).glob("*/*/*.wav")):
        # name = str(path).split('/')[-1].split('.')[0]
        label = str(path).split('_')[-4]
        try:
            # There are some broken files
            s = torchaudio.load(path)
            data.append({
                # "name": name,
                "id": path,
                "label": label
            })
        except Exception as e:
            # print(str(path), e)
            pass
    return data
def preprocess_Parkinson_csv(data_path,save_path):
    data = generate_csv_from_path(data_path)
    df = pd.DataFrame(data)
    df.replace('KCL/SpontaneousDialogue/HC/ID22hc', 'hc', inplace=True)
    # df.to_csv(data_path + 'Parkinson.csv', index=False)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_df, valid_df = train_test_split(train_df,test_size=0.1, random_state=42,stratify=train_df['label'])


    train_df.to_csv(f"{save_path}/Parkinsontrain.csv", encoding="utf-8", index=False)
    valid_df.to_csv(f"{save_path}/Parkinsonvalid.csv", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/Parkinsontest.csv", encoding="utf-8", index=False)

if __name__ == '__main__':
    print('This only works for KCL Parkinson dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", default=None, type=str, required=True)  # Path of data
    parser.add_argument("--save_path", default=None, type=str, required=True)  # Path to save the csv file
    args = parser.parse_args()
    data_path = args.load_path
    save_path = args.save_path
    preprocess_Parkinson_csv(data_path,save_path)

