import torch
import torch.nn as nn
import sys
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evalute_best_model, add_linear
from Models.autoencoder import cal
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#emotion
csv_path = '/home/ubuntu/capstone/Data/'
train_df = pd.read_csv(csv_path + "emotion_train.csv")
train_df = train_df[['emotion', "Image_file_path"]]
train_df.columns=['label','id']

train_df['label'] = manual_label_encoder(train_df['label'],'emotion')


test_df = pd.read_csv(csv_path + "emotion_test.csv")
test_df = test_df[['emotion', "Image_file_path"]]
test_df.columns=['label','id']

test_df['label'] = manual_label_encoder(test_df['label'],'emotion')


#race
# train_df = pd.read_csv("race_train.csv")
# test_df = pd.read_csv("race_test.csv")

classes = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

batch_size = 64
epochs = 5
IMAGE_SIZE = 128
num_layers = 3
learning_rate = 0.00002
OUTPUTS_a = len(classes)

train_loader = dataloader(train_df, OUTPUTS_a = 6, BATCH_SIZE = 64, IMAGE_SIZE=128)
test_loader = dataloader(test_df, OUTPUTS_a = 6, BATCH_SIZE = 64, IMAGE_SIZE=128)

jj, kk = cal(IMAGE_SIZE, num_layers)
cnn = CNN(encoded_space_dim=batch_size, jj=jj, kk=kk)
cnn = add_linear(cnn, OUTPUTS_a)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

train_and_test(cnn, train_loader, test_loader, classes, num_layers, epochs, batch_size, learning_rate)

evalute_best_model(cnn, test_loader, classes, num_layers)