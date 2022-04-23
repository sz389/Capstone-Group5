import torch
import torch.nn as nn
import sys
from torchvision import models
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evaluate_best_model, add_linear, pretrained_model, CNN9
from Models.autoencoder import cal
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder, get_model_params, compute_metrics
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Race
csv_path = '/home/ubuntu/capstone/Data/'
train_df = pd.read_csv(csv_path + "race_train.csv")
train_df = train_df[['Race', "Image_file_path"]]
train_df.columns=['label','id']

train_df['label'] = manual_label_encoder(train_df['label'],'race')

val_df = pd.read_csv(csv_path + "race_val.csv")
val_df = val_df[['Race', "Image_file_path"]]
val_df.columns=['label','id']

val_df['label'] = manual_label_encoder(val_df['label'],'race')

test_df = pd.read_csv(csv_path + "race_test.csv")
test_df = test_df[['Race', "Image_file_path"]]
test_df.columns=['label','id']

test_df['label'] = manual_label_encoder(test_df['label'],'race')

classes = ['Caucasian', 'African American', 'Asian']
OUTPUTS_a = len(classes)

batch_size = 64
d = 64
epochs = 30
IMAGE_SIZE = 128
num_layers = 3
learning_rate = 1e-3

train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
val_loader = dataloader(val_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

#jj, kk = cal(IMAGE_SIZE, num_layers)
#cnn = CNN(d, jj, kk, OUTPUTS_a)
cnn = pretrained_model("vgg16", OUTPUTS_a)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

#model_name = 'cnn_{}_layers_emotions.pt'.format(num_layers)
model_name = "vgg16_race_with_val.pt"
train_and_test(cnn, train_loader, val_loader, classes, model_name, epochs, batch_size, learning_rate)
evaluate_best_model(cnn, test_loader, classes, model_name)
