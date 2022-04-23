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
from Wav2vec import dataset
#%%
model1 = AutoModelForAudioClassification.from_pretrained("/home/ubuntu/Capstone/saved_model/wav2vec2-base-finetuned-ks/checkpoint-90/")
#%%
df_test = pd.read_csv("/home/ubuntu/Capstone/data/KCL_test_trim_split_audio.csv")
test_set = dataset(df_test)
#%%
speech, sr = librosa.load(df_test.id[0],sr=16000)
speech = torch.Tensor(speech)
#%%
model_checkpoint = "facebook/wav2vec2-base"
model_name = model_checkpoint.split("/")[-1]
OUTPUTS_a =2
labels = ['hc','pd']
model_path = "/home/ubuntu/Capstone/saved_model/"
args = TrainingArguments(
    model_path+f"{model_name}-finetuned-ks1",
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
metric = load_metric("accuracy",'f1')
feature_extractor1 = AutoFeatureExtractor.from_pretrained(
    "/home/ubuntu/Capstone/saved_model/wav2vec2-base-finetuned-ks/checkpoint-90/")
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
trainer = Trainer(
    model1,
    args,
    # train_dataset=trainset,
    # eval_dataset=testset,
    tokenizer=feature_extractor1,
    compute_metrics=compute_metrics
)
#%%

# predicts = trainer.predict(test_set)
# #%%
# predict12 = predicts[1]
#%%
# _, predicted = np.max(predicts.predictions, 1)
#%%
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return 1

    def __getitem__(self,idx):
        return {'input_values':self.tokenized_texts}


test_dataset = SimpleDataset(speech)
#%%
xi = trainer.predict(test_dataset)
print('\n')
print(f'prediction: {np.argmax(xi.predictions)}')