import numpy as np
import torch
import torch.nn as nn
import sys
import pandas as pd
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.autoencoder import Encoder, Decoder, cal
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#emotion
csv_path = '/home/ubuntu/capstone/Data/'
train_df = pd.read_csv(csv_path + "race_train.csv")
train_df = train_df[['Race', "Image_file_path"]]
train_df.columns=['label','id']

train_df['label'] = manual_label_encoder(train_df['label'],'Race')

test_df = pd.read_csv(csv_path + "race_test.csv")
test_df = test_df[['Race', "Image_file_path"]]
test_df.columns=['label','id']

test_df['label'] = manual_label_encoder(test_df['label'],'Race')

batch_size = 64
epochs = 200
lr = 1e-3
d = 64

IMAGE_SIZE = 128
num_layers = 3
OUTPUTS_a = 6

train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

jj, kk = cal(IMAGE_SIZE, num_layers)
encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)
criterion = nn.MSELoss()

print("Starting Autoencoder...")

for epoch in range(epochs):
    encoder.train()
    decoder.train()
    loss = []
    for batch_features, label in train_loader:
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        encoded_data = encoder(batch_features.float())
        decoded_data = decoder(encoded_data)
        train_loss = criterion(decoded_data, batch_features.float())
        train_loss.backward()
        optimizer.step()
        loss.append(train_loss.detach().cpu().numpy().item())
    losses = np.mean(loss)

    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, losses))

PATH_SAVE = "/home/ubuntu/capstone/CNN/Models/Saved_Models/"
torch.save(encoder.state_dict(), PATH_SAVE + 'encoder_{}_layers.pt'.format(num_layers))
torch.save(decoder.state_dict(), PATH_SAVE + 'decoder.{}_layers.pt'.format(num_layers))

