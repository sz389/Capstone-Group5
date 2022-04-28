#%%
import pandas as pd
import os
from Models.cnn import train_and_test,CNN,evaluate_best_model,pretrained_model,combine_and_evaluate
from utility import Dataset,dataloader,get_n_params
#%%
def process_df_path(df):
    path1 = '/home/ubuntu/Capstone/data_train_val_test/new_mel_spectrograms/'
    df['id'] = df['path']
    return df

#%%
import argparse
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    parser.add_argument("--model_dir", default=None, type=str, required=True)
    parser.add_argument("--train_csv", default=None, type=str, required=True)
    parser.add_argument("--test_csv", default=None, type=str, required=True)
    parser.add_argument("--val_csv", default=None, type=str, required=True)


    args = parser.parse_args()

    csv_path1 = args.csv_load_path
    model_dir = args.model_dir
    train_csv = args.train_csv
    test_csv = args.test_csv
    val_csv = args.val_csv

   # csv_path1 = '/home/ubuntu/Capstone/data_train_val_test/'
    df_train = pd.read_csv(csv_path1 + train_csv)
    df_test = pd.read_csv(csv_path1 + test_csv)
    df_val = pd.read_csv(csv_path1 + val_csv)

    df_train['label'] = df_train['native_language']
    df_val['label'] = df_val['native_language']
    df_test['label'] = df_test['native_language']



    df_train,df_test, df_val = process_df_path(df_train),process_df_path(df_test), process_df_path(df_val)
  #  model_dir = "/home/ubuntu/Capstone/CNN/Models/Saved_Models/new_saved_models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    OUTPUTS_a = 5
    batch_size = 16
    classnames = ['arabic', 'english', 'french', 'mandarin', 'spanish']
    model_savename = model_dir+'resnet34.pt'
    train_loader = dataloader(xdf_dset=df_train,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=True)
    val_loader = dataloader(xdf_dset=df_val, OUTPUTS_a=OUTPUTS_a,
                             BATCH_SIZE= batch_size, IMAGE_SIZE=128, shuffle=True)
    test_loader = dataloader(xdf_dset=df_test,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=False)

    model = pretrained_model('resnet34',OUTPUTS_a=OUTPUTS_a)
    train_and_test(cnn=model,train_loader=train_loader,test_loader=val_loader,
                   classes=classnames,model_savename=model_savename,
                   num_epochs = 20, batch_size = batch_size, learning_rate = 1E-3)
    # evalute_best_model(cnn=model,model_path=model_savename,test_loader=test_loader,
    #                    classes=classnames)
    combine_and_evaluate(cnn=model,model_path=model_savename,test_loader=test_loader,
                         xdf_dset_test=df_test,class_names=classnames)
    get_n_params(model)
