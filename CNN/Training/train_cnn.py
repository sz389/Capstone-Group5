import torch
import torch.nn as nn
from CNN.Models.cnn import CNN, train_and_test, evalute_best_model, add_linear
from CNN.Models.autoencoder import cal
from CNN.Utility.dataloader import dataloader
from CNN.Utility.utility import manual_label_encoder
import pandas as pd
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("/home/ubuntu/capstone/Data/age_train.csv")
train_df = train_df[["Age_by_decade","Image_file_path"]]
train_df.columns=["label","id"]
train_df["label"] = manual_label_encoder(train_df['label'],"age")

test_df = pd.read_csv("/home/ubuntu/capstone/Data/age_test.csv")
test_df = test_df[["Age_by_decade","Image_file_path"]]
test_df.columns=["label","id"]
test_df["label"] = manual_label_encoder(test_df['label'],"age")

classes = [ '<30s', '30s', '40s', '50s','>60s']

batch_size = 64
epochs = 5
IMAGE_SIZE = 128
num_layers = 3
learning_rate = 0.00002
OUTPUTS_a = len(classes)

train_loader= dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = 64, IMAGE_SIZE=128)
test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = 64, IMAGE_SIZE=128)

jj, kk = cal(IMAGE_SIZE, num_layers)
cnn = CNN(encoded_space_dim=batch_size, jj=jj, kk=kk, OUTPUTS_a = OUTPUTS_a)
cnn = add_linear(cnn,OUTPUTS_a)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

train_and_test(cnn, train_loader, test_loader, classes, num_layers, epochs, batch_size, learning_rate)

evalute_best_model(cnn, test_loader, classes, num_layers)
