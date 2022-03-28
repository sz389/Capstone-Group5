#%%
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer


from pathlib import Path
from tqdm import tqdm
from transformers import AutoFeatureExtractor
from sklearn.preprocessing import LabelEncoder

import torchaudio
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import os
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
import librosa
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils import data
#%%
os.chdir('..')
csv_path = os.getcwd()+'/26-29_09_2017_KCL/'
model_path = csv_path +'saved_transformer/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
# df = pd.read_csv(csv_path+'KCL_trim_split_spec.csv')
# df = df[['label','wav_name']]
# df.columns =['label','id']
# df['id'] = csv_path+'trimed_audio/'+df['id']
#%%
df_train,df_test = pd.read_csv(csv_path+'/trainset4trans.csv'),\
                    pd.read_csv(csv_path+'/testset4trans.csv')
# data_files = {
#     "train": csv_path + "KCL_split_train.csv",
#     "validation": csv_path + "KCL_split_test.csv",
# }
#
# dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
# train_dataset = dataset["train"]
# eval_dataset = dataset["validation"]
#%%
df = pd.concat([df_train,df_test])
#%%
model_checkpoint = "facebook/wav2vec2-base"
OUTPUTS_a =2
def process_target():
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''
    xtarget = list(np.array(df['label'].unique()))
    le = LabelEncoder()
    le.fit(xtarget)
    final_target = le.transform(np.array(df['label']))
    xtarget.reverse()
    class_names=xtarget
    df['label'] = final_target
    return class_names
labels = process_target()
df_train,df_test = df.iloc[:df_train.shape[0],:],df.iloc[df_train.shape[0]:,:]

class dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, df, transform=None):
        # Initialization
        self.df = df
        self.transform = transform
    def __len__(self):
        # Denotes the total number of samples'
        return len(self.df)
    def __getitem__(self, index):
        # Generates one sample of data'
        # Select sample
        # Load data and get label
        y=self.df.label.iloc[index]
        # labels_ohe = np.zeros(OUTPUTS_a)
        # for idx, label in enumerate(range(OUTPUTS_a)):
        #     if label == y:
        #         labels_ohe[idx] = 1
        # y = torch.FloatTensor(labels_ohe)
        file_name = self.df.path.iloc[index]
        X,sr = librosa.load(file_name,sr=feature_extractor.sampling_rate)
        # X = feature_extractor(
        #     X,
        #     # sampling_rate=feature_extractor.sampling_rate,
        #     # max_length=int(feature_extractor.sampling_rate * max_duration),
        #     # truncation=True,
        # )
        dict = {'input_values':X,'label':y}
        return dict
#%%
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["label"])
metric = load_metric("accuracy",'f1')
trainset = dataset(df_train)
testset = dataset(df_test)
#%%
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
#%%
batch_size = 4
num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    model_path+f"{model_name}-finetuned-ks",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    # per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    # per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
trainer = Trainer(
    model,
    args,
    train_dataset=trainset,
    eval_dataset=testset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)
#%%
trainer.train()

