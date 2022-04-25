#%%
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from CNN.models.cnn import train_and_test,CNN,CNN9,define_cnn,evaluate_best_model,pretrained_model,combine_and_evaluate
from utility import Dataset,dataloader,get_n_params
import argparse
#%%

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder", default=None, type=str, required=True)  # Path of csv to load
    parser.add_argument("--model_dir", default=None, type=str, required=True)  # Path to save the csv file
    args = parser.parse_args()
    csv_path = args.csv_folder
    model_dir = args.model_dir
    # csv_path = "/home/ubuntu/Capstone/data/"
    # model_dir = "/home/ubuntu/Capstone/saved_model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    df_train = pd.read_csv(csv_path+"/KCL_train_trim_split_spec.csv")
    df_val = pd.read_csv(csv_path+"/KCL_valid_trim_split_spec.csv")
    df_test = pd.read_csv(csv_path+"/KCL_test_trim_split_spec.csv")
    classnames = ['hc','pd']
    OUTPUTS_a = len(classnames)
    batch_size = 16
    num_epochs = 50
    train_loader = dataloader(xdf_dset=df_train,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=True)
    val_loader = dataloader(xdf_dset=df_val,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=True)
    test_loader = dataloader(xdf_dset=df_test,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=False)

    # if using pretrained model: resnet34,resnet18,efficientnet_b2
    # model = pretrained_model('efficientnet_b2',OUTPUTS_a=OUTPUTS_a)
    # model_savename = model_dir + 'efficientnet_b2.pt'
    # if using defined cnn: cnn9
    model = define_cnn(OUTPUTS_a=len(classnames))
    model_savename = model_dir + 'cnn9_parkinson.pt'

    train_and_test(cnn=model,train_loader=train_loader,test_loader=val_loader,
                   classes=classnames,model_savename=model_savename,
                   num_epochs = num_epochs, batch_size = batch_size, learning_rate = 1E-3)
    evaluate_best_model(cnn=model,model_path=model_savename,test_loader=test_loader,
                       classes=classnames)
    combine_and_evaluate(cnn=model,model_path=model_savename,test_loader=test_loader,
                         xdf_dset_test=df_test,class_names=classnames)
    print(f'model parameters: {get_n_params(model)}')
