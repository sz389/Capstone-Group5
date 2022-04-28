#%%
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor
import os
import librosa
from torch.utils import data
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
        file_name = self.df.path.iloc[index]
        X,sr = librosa.load(file_name,sr=feature_extractor.sampling_rate)
        dict = {'input_values':X,'label':y}
        return dict
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

#%%
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)  # path to load in csv
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument("--train_csv", default=None, type=str, required=True)  # train_csv
    parser.add_argument("--val_csv", default=None, type=str, required=True)  # val_csv

    args = parser.parse_args()
    model_path = args.model_path
    csv_path = args.csv_load_path
    train_csv = args.train_csv
    val_csv = args.val_csv


    if not os.path.exists(model_path):
        os.makedirs(model_path)
    df_train,df_test = pd.read_csv(csv_path + train_csv),\
                        pd.read_csv(csv_path + val_csv)
    model_checkpoint = "facebook/wav2vec2-base"
    labels =  ['arabic', 'english', 'french', 'mandarin', 'spanish']
    OUTPUTS_a =len(labels)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    metric = load_metric("accuracy",'f1')
    trainset = dataset(df_train)
    testset = dataset(df_test)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

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
        model_path+f"{model_name}-finetuned",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=3e-5,
        # per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        # per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
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
