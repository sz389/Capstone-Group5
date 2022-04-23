import torch
import sys
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evaluate_best_model, add_linear
from Models.autoencoder import cal, Encoder, Decoder
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder
import pandas as pd
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#emotion
csv_path = '/home/ubuntu/capstone/Data/'
train_df = pd.read_csv(csv_path + "race_train.csv")
train_df = train_df[['emotion', "Image_file_path"]]
train_df.columns=['label','id']

train_df['label'] = manual_label_encoder(train_df['label'],'race')

test_df = pd.read_csv(csv_path + "race_test.csv")
test_df = test_df[['race', "Image_file_path"]]
test_df.columns=['label','id']

test_df['label'] = manual_label_encoder(test_df['label'],'race')

batch_size = 64
epochs = 5
lr = 1e-3
d = 64
IMAGE_SIZE = 128
num_layers = 3  # change based on autoencoder architecture
OUTPUTS_a = 6

train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

jj, kk = cal(IMAGE_SIZE, num_layers)
encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

PATH_SAVE = "/home/ubuntu/capstone/CNN/Models/Saved_Models/"
encoder.load_state_dict(torch.load(PATH_SAVE + 'encoder_{}_layers.pt'.format(num_layers)))
decoder.load_state_dict(torch.load(PATH_SAVE + 'decoder.{}_layers.pt'.format(num_layers)))

# params_to_optimize = [
#     {'params': encoder.parameters()},
#     {'params': decoder.parameters()}
# ]

print("Starting Autoencoder...")

#optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    encoder.eval()
    decoder.eval()
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

