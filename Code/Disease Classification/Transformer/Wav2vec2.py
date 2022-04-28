#%%
from datasets import load_metric
import numpy as np
import pandas as pd
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor
import os
import librosa
from torch.utils import data
from utility import get_n_params
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

#%%
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder", default=None, type=str, required=True)  # Path of csv to load
    parser.add_argument("--model_dir", default=None, type=str, required=True)  # Path to save the csv file
    args = parser.parse_args()
    csv_path = args.csv_folder
    model_path = args.model_dir
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # load the data
    df_train,df_test = pd.read_csv(csv_path+"/KCL_train_trim_split_audio.csv"),\
                        pd.read_csv(csv_path+"/KCL_valid_trim_split_audio.csv")
    labels = ['hc', 'pd']
    trainset = dataset(df_train)
    testset = dataset(df_test)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Define the model
    model_checkpoint = "facebook/wav2vec2-base"
    # model_checkpoint = "microsoft/wavlm-large"
    # model_checkpoint = "hf-internal-testing/tiny-random-unispeech-sat"
    # model_checkpoint = "facebook/wav2vec2-large-960h"
    OUTPUTS_a =len(labels)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    metric = load_metric("accuracy",'f1')

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
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
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
    print(f'Parameters: {get_n_params(model)}')