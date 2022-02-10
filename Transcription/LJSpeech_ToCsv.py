#%%
import os
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split
# import sys
#%%
os.chdir('/home/ubuntu/Capstone/')
csv_path = os.getcwd()+'/LJSpeech-1.1/'
# csv_path = "/home/ubuntu/Capstone/LJSpeech-1.1/metadata.csv"
transcript = pd.read_csv(csv_path+'metadata.csv',sep = '|',header=None)
#%%
transcript.columns = ['filename','text','text1']
transcript['filename'] += '.wav'
file_path = csv_path + 'wavs/'
transcript['path'] = file_path+transcript['filename']
print(f"Step 0: {len(transcript)}")
transcript["status"] = transcript["path"].apply(lambda path: True if os.path.exists(path) else None)
transcript = transcript.dropna(subset=["path"])
transcript = transcript.drop("status", 1)
print(f"Step 1: {len(transcript)}")
#%%
transcript = transcript[['filename','path','text']]
transcript.to_csv(csv_path+'LJSpeech.csv',index=False)
#%%
df  = pd.read_csv(csv_path+'/LJSpeech.csv',index=False)
#%%
save_path = csv_path

train_df, test_df = train_test_split(df, test_size=0.2, random_state=101)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)


print(train_df.shape)
print(test_df.shape)