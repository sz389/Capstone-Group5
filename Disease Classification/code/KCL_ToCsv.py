#%%
import os
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split
#%%
os.chdir('..')
data_path = os.getcwd()+'/26-29_09_2017_KCL/'
#%%
data = []
for path in tqdm(Path(data_path).glob("*/*/*.wav")):
    # name = str(path).split('/')[-1].split('.')[0]
    label = str(path).split('_')[-4]

    try:
        # There are some broken files
        s = torchaudio.load(path)
        data.append({
            # "name": name,
            "path": path,
            "Disease": label
        })
    except Exception as e:
        # print(str(path), e)
        pass
#%%
df = pd.DataFrame(data)
print(df.head())
#%%
df.replace('KCL/SpontaneousDialogue/HC/ID22hc','hc',inplace=True)
print("Labels: ", df["Disease"].unique())
print(df.groupby("Disease").count()[["path"]])
#%%
df.to_csv(data_path+'Parkinson.csv',index=False)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["Disease"])
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(f"{data_path}/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{data_path}/test.csv", sep="\t", encoding="utf-8", index=False)