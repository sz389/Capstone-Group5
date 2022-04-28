#%%
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,classification_report
from transformers import AutoFeatureExtractor
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_load_path", default=None, type=str, required=True)  # path to load in csv
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument("--test_csv", default=None, type=str, required=True)  # train_csv
    parser.add_argument("--model_name", default=None, type=str, required=True)
    args = parser.parse_args()

    csv_path = args.csv_load_path
    model_path = args.model_path
    test_csv = args.test_csv
    model_name = args.model_name


    # define the model informatio
    #best_model_path = model_path+"/wav2vec2-base-finetuned-ks/checkpoint-868/"
    #best_model_path = model_path+"wav2vec2-large-960h-finetuned-ks/checkpoint-1240"
    best_model_path = model_path +'/'+ model_name
    model1 = AutoModelForAudioClassification.from_pretrained(best_model_path)
    metric = load_metric("accuracy",'f1')
    feature_extractor = AutoFeatureExtractor.from_pretrained(best_model_path)
    model_checkpoint = "facebook/wav2vec2-base"
    model_name = model_checkpoint.split("/")[-1]
    # define the data and dataset
    df_test = pd.read_csv(csv_path + test_csv)
    test_set = dataset(df_test)
    OUTPUTS_a = 5
    labels = ['arabic', 'english', 'french', 'mandarin', 'spanish']

    args = TrainingArguments(
        model_path+f"{model_name}-finetuned-ks1",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=3e-5,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    trainer = Trainer(
        model1,
        args,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )
    #  evaluate on test set
    print('='*80)
    print('results on chunks')
    predict_label = trainer.predict(test_set)
    predictions = np.argmax(predict_label.predictions,1)
    f1 = f1_score(df_test['label_num'], predictions, average='micro')
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
    # Combine the chunks according to original file to get predictions
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
        label_list.append(subset['native_language'].iloc[0])
        label_num_list.append(subset['label_num'].iloc[0])
    pred_result = pd.DataFrame()
    pred_result['origin'] = origin_list
    pred_result['label'] = label_list
    pred_result['label_num'] = label_num_list
    pred_result['prediction'] = list_pred
    cf_matrix = confusion_matrix(pred_result['label_num'], pred_result['prediction'])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in labels],
                         columns=[i for i in labels])
    print('classification report: ')
    print(classification_report(pred_result['label_num'], pred_result['prediction'], target_names=labels))
    print('Confusion matrix: ')
    print(df_cm)
    f1 = f1_score(pred_result['label_num'], pred_result['prediction'],average='micro')
    print(f'final f1 score:{f1}')
    print(f'Accuracy: {accuracy_score(pred_result.label_num, pred_result.prediction)}')
    pred_result.to_csv('transformer_result.csv', index=False)
    print('=' * 80)
