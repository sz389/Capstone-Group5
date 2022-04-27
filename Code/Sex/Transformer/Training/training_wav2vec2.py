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
import sys
import argparse
sys.path.insert(1, '/home/ubuntu/capstone/Transformer')
from Utility.utility import manual_label_encoder, get_classes
#%%
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
        y=self.df.label.iloc[index]
        file_name = self.df.id.iloc[index]
        X,sr = librosa.load(file_name,sr=feature_extractor.sampling_rate)
        dict = {'input_values':X,'label':y}
        return dict

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True) #path to load in csv
    parser.add_argument('--category', default=None, type=str, required=True) #category (Ex. emotion, race, sex, age)
    parser.add_argument("--model_save_name", default=None, type=str, required=True)
    parser.add_argument("--model_save_path", default=None, type=str, required=True)
    parser.add_argument("--train_csv", default=f"sex_train.csv", type=str, required=False)  # train_csv
    parser.add_argument("--val_csv", default=f"sex_val.csv", type=str, required=False)  # val_csv
    parser.add_argument("--epochs", default=20, type=int, required=False)
    parser.add_argument("--batch_size", default=4, type=int, required=False)
    parser.add_argument("--learning_rate", default=3e-5, type=int, required=False)

    args = parser.parse_args()
    category = args.category
    csv_load_path = args.csv_load_path
    train_csv = args.train_csv
    val_csv = args.val_csv
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    model_save_name = args.model_save_name
    model_path = args.model_save_path

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    df_train,df_test = pd.read_csv(csv_load_path + train_csv),\
                        pd.read_csv(csv_load_path + val_csv)

    df_train[category] = manual_label_encoder(df_train[category], category)

    df_train = df_train[[category, "Audio_file_path"]]
    df_train.columns = ['label', 'id']

    df_test[category] = manual_label_encoder(df_test[category], category)

    df_test = df_test[[category, "Audio_file_path"]]
    df_test.columns = ['label', 'id']

    model_checkpoint = "facebook/wav2vec2-base"

    labels = get_classes(category)
    OUTPUTS_a =len(labels)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    metric = load_metric("accuracy",'f1')
    trainset = dataset(df_train)
    testset = dataset(df_test)

    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        model_path+f"{model_name}-{model_save_name}",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )
    trainer.train()
