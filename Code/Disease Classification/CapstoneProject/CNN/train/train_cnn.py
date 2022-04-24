#%%
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from CNN.models.cnn import train_and_test,CNN,evalute_best_model,pretrained_model,combine_and_evaluate
from utility import Dataset,dataloader,get_n_params
#%%

if __name__ =='__main__':
    df_train = pd.read_csv("/home/ubuntu/Capstone/data/KCL_train_trim_split_spec.csv")
    df_test = pd.read_csv("/home/ubuntu/Capstone/data/KCL_test_trim_split_spec.csv")
    model_dir = "/home/ubuntu/Capstone/saved_model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    OUTPUTS_a = 2
    classnames = ['hc','pd']
    model_savename = model_dir+'resnet34.pt'
    train_loader = dataloader(xdf_dset=df_train,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = 64, IMAGE_SIZE=128,shuffle=True)
    test_loader = dataloader(xdf_dset=df_test,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = 64, IMAGE_SIZE=128,shuffle=False)
    model = pretrained_model('resnet34',OUTPUTS_a=OUTPUTS_a)
    train_and_test(cnn=model,train_loader=train_loader,test_loader=test_loader,
                   classes=classnames,model_savename=model_savename,
                   num_epochs = 20, batch_size = 64, learning_rate = 1E-3)
    evalute_best_model(cnn=model,model_path=model_savename,test_loader=test_loader,
                       classes=classnames)
    combine_and_evaluate(cnn=model,model_path=model_savename,test_loader=test_loader,
                         xdf_dset_test=df_test,class_names=classnames)
    get_n_params(model)
