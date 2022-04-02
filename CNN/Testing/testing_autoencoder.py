import torch
from torch import nn
from autoencoder import Encoder, Decoder, cal
from dataloader import dataloader
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("...")
test_df = pd.read_csv("...")

train_loader, test_loader = dataloader(train_df, test_df, OUTPUTS_a = 6, BATCH_SIZE = 64, IMAGE_SIZE=128)
#xdf_dset, xdf_dset_test, OUTPUTS_a, BATCH_SIZE = 64, IMAGE_SIZE=128

batch_size = 64
epochs = 5
lr = 1e-3
d = 64
IMAGE_SIZE = 128
num_layers = 3 #change based on autoencoder architecture

jj, kk = cal(IMAGE_SIZE, num_layers)
encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

PATH_SAVE = "/home/ubuntu/capstone/Models/Saved_Models/"
encoder.load_state_dict(torch.load(PATH_SAVE + 'encoder_{}_layers.pt'.format(num_layers)))
decoder.load_state_dict(torch.load(PATH_SAVE + 'decoder.{}_layers.pt'.format(num_layers)))

# params_to_optimize = [
#     {'params': encoder.parameters()},
#     {'params': decoder.parameters()}
# ]

print("Starting Autoencoder...")

#optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)
criterion = nn.MSEloss()

for epoch in range(epochs):
    encoder.train()
    decoder.train()
    loss = []
    for batch_features, label in train_loader:
        batch_features = batch_features.to(device)
        #optimizer.zero_grad()
        encoded_data = encoder(batch_features.float())
        decoded_data = decoder(encoded_data)
        train_loss = criterion(decoded_data, batch_features.float())
        train_loss.backward()
        #optimizer.step()
        loss.append(train_loss.detach().cpu().numpy().item())
    losses = np.mean(loss)

    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, losses))

