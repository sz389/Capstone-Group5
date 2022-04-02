import os
import numpy as np
import torch
import torch.nn as nn
from autoencoder import Encoder, Decoder, cal
from dataloader import dataloader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("...")
train_loader = dataloader(train_df)

model_name = 'autoencoder'
csv_path = os.getcwd()
model_path = csv_path +'/saved_autoencoder/'
if not os.path.exists(model_path):
    os.makedirs(model_name)

batch_size = 64
epochs = 5
lr = 1e-3
d = 64
IMAGE_SIZE  = 128

max_sent = IMAGE_SIZE
jj, kk = cal(max_sent)
encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

print("Starting Autoencoder...")

optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)
criterion = nn.MSELoss()

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

torch.save(encoder.state_dict(), PATH_SAVE + 'encoder.pt')
torch.save(decoder.state_dict(), PATH_SAVE + 'decoder.pt')


