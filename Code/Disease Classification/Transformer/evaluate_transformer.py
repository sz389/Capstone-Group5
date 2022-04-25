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
from Wav2vec import compute_metrics
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
        y=self.df.label_num.iloc[index]
        file_name = self.df.id.iloc[index]
        X,sr = librosa.load(file_name,sr=feature_extractor.sampling_rate)
        dict = {'input_values':X,'label':y}
        return dict
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

if __name__ == '__main__':
    # define the model information
    model_path = "/home/ubuntu/Capstone/saved_model/"
    best_model_path = model_path+"/wav2vec2-base-finetuned-ks/checkpoint-150/"
    # best_model_path = model_path + "tiny-random-unispeech-sat-finetuned-ks/checkpoint-30"
    # best_model_path = model_path+"wav2vec2-large-960h-finetuned-ks/checkpoint-150"
    model1 = AutoModelForAudioClassification.from_pretrained(best_model_path)
    metric = load_metric("accuracy",'f1')
    feature_extractor = AutoFeatureExtractor.from_pretrained(best_model_path)
    model_checkpoint = "facebook/wav2vec2-base"
    model_name = model_checkpoint.split("/")[-1]
    # define the data and dataset
    df_test = pd.read_csv("/home/ubuntu/Capstone/data/KCL_test_trim_split_audio.csv")
    test_set = dataset(df_test)
    OUTPUTS_a =2
    labels = ['hc','pd']

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
    f1 = f1_score(df_test['label_num'], predictions)
    df_test['prediction'] = predictions
    cf_matrix = confusion_matrix(df_test['label_num'], predictions)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in labels],
                         columns=[i for i in labels])
    print('classification report: ')
    print(classification_report(df_test['label_num'], predictions, target_names=labels))
    print('Confusion matrix: ')
    print(df_cm)
    print(f'F1 score: {f1}')
    acc = accuracy_score(df_test['label_num'], predictions)
    print(f'Accuracy: {acc}')
    # Combine the chunks according to original file to get predicitons
    print('='*80)
    print('results after combining to origin')
    unique_origin = df_test.origin.unique()
    list_pred = []
    label_list = []
    origin_list = []
    label_num_list = []
    for i, ori in enumerate(unique_origin):
        subset = df_test[df_test['origin'] == ori]
        list_pred.append(1 if subset.prediction.mean() > 0.5 else 0)
        origin_list.append(ori)
        label_list.append(subset['label'].iloc[0])
        label_num_list.append(subset['label_num'].iloc[0])
    pred_result = pd.DataFrame()
    pred_result['origin'] = origin_list
    pred_result['label'] = label_list
    pred_result['label_num'] = label_num_list
    pred_result['prediction'] = list_pred
    # 0: hc(healthy control) 1: pd(parkinson disease)
    cf_matrix = confusion_matrix(pred_result['label_num'], pred_result['prediction'])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in labels],
                         columns=[i for i in labels])
    print('classification report: ')
    print(classification_report(pred_result['label_num'], pred_result['prediction'], target_names=labels))
    print('Confusion matrix: ')
    print(df_cm)
    f1 = f1_score(pred_result['label_num'], pred_result['prediction'])
    print(f'final f1 score:{f1}')
    print(f'Accuracy: {accuracy_score(pred_result.label_num, pred_result.prediction)}')
    pred_result.to_csv('result.csv', index=False)
    print('=' * 80)

