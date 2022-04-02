import numpy as np
import torch
import torch.nn as nn
from autoencoder import cal
from cnn import CNN, train_and_test, evalute_best_model
from dataloader import dataloader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("...")
test_df = pd.read_csv("...")

train_loader, test_loader = dataloader(train_df, test_df, OUTPUTS_a = 6, BATCH_SIZE = 64, IMAGE_SIZE=128)

batch_size = 64
epochs = 5
IMAGE_SIZE = 128
num_layers = 3
learning_rate = 0.00002

classes = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
OUTPUTS_a = len(classes)

jj, kk = cal(IMAGE_SIZE, num_layers)
cnn = CNN(encoded_space_dim=batch_size, jj=jj, kk=kk, OUTPUTS_a = OUTPUTS_a)

PATH_SAVE = "/home/ubuntu/capstone/Models/Saved_Models/"
cnn.load_state_dict(torch.load(PATH_SAVE + 'encoder_{}_layers.pt'.format(num_layers)))

train_and_test(cnn, train_loader, test_loader, classes, num_layers, epochs, batch_size, learning_rate)
evalute_best_model(cnn, test_loader, num_layers)
