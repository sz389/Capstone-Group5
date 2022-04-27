#%%
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,classification_report
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
import argparse
import sys
sys.path.insert(1, '/home/ubuntu/capstone/Transformer')
from Utility.utility import manual_label_encoder, get_classes

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)  # path to load in csv
    parser.add_argument('--category', default=None, type=str, required=True)  # category (Ex. emotion, race, sex, age)
    parser.add_argument("--model_load_path", default=None, type=str, required=True)
    parser.add_argument("--model_dir_path", default=None, type=str, required=True)
    parser.add_argument("--test_csv", default=f"emotion_test.csv", type=str, required=False)  # test_csv
    parser.add_argument("--epochs", default=20, type=int, required=False)
    parser.add_argument("--batch_size", default=4, type=int, required=False)
    parser.add_argument("--learning_rate", default=3e-5, type=int, required=False)

    args = parser.parse_args()
    category = args.category
    csv_load_path = args.csv_load_path
    model_load_path = args.model_load_path
    model_dir_path = args.model_dir_path
    test_csv = args.test_csv
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # define the model information
    model1 = AutoModelForAudioClassification.from_pretrained(model_load_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_load_path)
    metric = load_metric("accuracy",'f1')

    model_checkpoint = "facebook/wav2vec2-base"
    model_name = model_checkpoint.split("/")[-1] #wav2vec2-base

    # define the data and dataset
    df_test = pd.read_csv(csv_load_path + test_csv)
    df_test[category] = manual_label_encoder(df_test[category], category)

    labels = get_classes(category)
    OUTPUTS_a =len(labels)

    df_test = df_test[[category, "Audio_file_path"]]
    df_test['id'] = df_test['Audio_file_path']
    df_test['label'] = df_test[category]
    test_set = dataset(df_test)

    model_dir = model_load_path.split("/")[0:2]

    args = TrainingArguments(
        model_dir_path,
        #"/home/ubuntu/capstone/Transformer/Models/Saved_Models/Transformer/",
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
        model1,
        args,
        # train_dataset=trainset,
        # eval_dataset=testset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )
    #  evaluate on test set
    print('='*80)
    print('results on chunks')
    predict_label = trainer.predict(test_set)
    predictions = np.argmax(predict_label.predictions,1)
    f1 = f1_score(df_test[category], predictions, average="weighted")
    df_test['prediction'] = predictions
    cf_matrix = confusion_matrix(df_test[category], predictions)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in labels],
                         columns=[i for i in labels])
    print('classification report: ')
    print(classification_report(df_test[category], predictions, target_names=labels))
    print('Confusion matrix: ')
    print(df_cm)
    print(f'F1 score: {f1}')
    acc = accuracy_score(df_test[category], predictions)
    print(f'Accuracy: {acc}')
