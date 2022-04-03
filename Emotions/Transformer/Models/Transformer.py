import numpy as np
import pandas as pd
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sys
sys.path.insert(1, '/home/ubuntu/capstone/Transformer')
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder

csv_path = '/home/ubuntu/capstone/Data/'
train_df = pd.read_csv(csv_path + "emotion_train.csv")
train_df = train_df[['emotion', "Audio_file_path"]]
train_df.columns=['label','id']

train_df['label'] = manual_label_encoder(train_df['label'],'emotion')


test_df = pd.read_csv(csv_path + "emotion_test.csv")
test_df = test_df[['emotion', "Audio_file_path"]]
test_df.columns=['label','id']

test_df['label'] = manual_label_encoder(test_df['label'],'emotion')

OUTPUTS_a = 6

labels = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

model_checkpoint = "facebook/wav2vec2-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

train_set = dataloader(train_df, feature_extractor)
test_set = dataloader(test_df, feature_extractor)

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
model_path = "/home/ubuntu/capstone/Transformer/Models/Saved_Models/"
args = TrainingArguments(
    model_path+f"{model_name}-finetuned-ks",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    eval_labels = eval_pred.label_ids
    predictions = np.argmax(eval_pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(eval_labels, predictions, average='micro')
    acc = accuracy_score(eval_labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


trainer = Trainer(
    model,
    args,
    train_dataset=train_set,
    eval_dataset=test_set,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

#%%
trainer.train()


#%%
trainer.evaluate() #Forest Gump -