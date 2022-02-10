#%%
import os
# os.system('sudo pip install torchaudio')
# os.system('sudo pip install librosa')
# os.system('sudo pip install Ipython')
# os.system('sudo pip install jiwer')
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split
import librosa

#%%
os.chdir('/home/ubuntu/Capstone/')
csv_path = os.getcwd()+'/LJSpeech-1.1/'
#%%
# import torchaudio
# import librosa
import IPython.display as ipd
# df  = pd.read_csv(csv_path+'/LJSpeech.csv',index=False)
#%%
from datasets import load_dataset, load_metric
data_files = {
    "train": csv_path + "train.csv",
    "validation": csv_path + "test.csv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print(train_dataset)
print(eval_dataset)
# We need to specify the input and output column
input_column = "path"
output_column = "text"
#%%
idx = np.random.randint(0, len(df))
sample = df.iloc[idx]
path = sample["path"]
text = sample["text"]
print(f"ID Location: {idx}")
print(f"      text: {text}")
print()
#%%
speech, sr = torchaudio.load(path)
speech = speech[0].numpy().squeeze()
speech = librosa.resample(np.asarray(speech), orig_sr = sr, target_sr=16_000)
#%%
ipd.Audio(data=np.asarray(speech), autoplay=True, rate=16000)
#%%

#%%
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
from jiwer import wer


model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
#%%
def map_to_array(batch):
    speech, _ = sf.read(batch["path"])
    batch["speech"] = speech
    return batch

dataset = dataset.map(map_to_array)

def map_to_pred(batch):
    input_values = processor(batch["speech"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch
#%%
# Look at one transcription
from transformers import Wav2Vec2Tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
input_values1 = tokenizer(speech, return_tensors = "pt").input_values
logits1 = model(input_values1.to('cuda')).logits
prediction1 = torch.argmax(logits1, dim = -1)
transcription1 = tokenizer.batch_decode(prediction1)[0]
print(transcription1)
print(f"      text: {text}")
#%%
result = dataset.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])
#%%
print("WER:", wer(result['train']["text"], result['train']["transcription"]))
print("WER:", wer(result['validation']["text"], result['validation']["transcription"]))