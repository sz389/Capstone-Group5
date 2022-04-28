import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
from Models.AutoEncoder import Encoder, Decoder, cal
from utility import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    parser.add_argument("--csv_file", default=None, type=str, required=True)
    parser.add_argument("--epochs", default=200, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--learning_rate", default=1e-3, type=int, required=False)
    parser.add_argument("--model_save_path", default=None, type=str, required=True)

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    csv_path = args.csv_load_path
    PATH_SAVE = args.model_save_path
    csv_file = args.csv_file

    classes = ['arabic', 'english', 'french', 'mandarin', 'spanish']
    OUTPUTS_a = len(classes)

    d = 64
    IMAGE_SIZE = 128
    num_layers = 3

    train_df = pd.read_csv(csv_path + '/' + csv_file)
    train_df['id'] = csv_path + 'mel_spectrograms/' + train_df['img_name']

    train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

    jj, kk = cal(IMAGE_SIZE, num_layers)
    encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
    decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
    criterion = nn.MSELoss()

    print("Starting Autoencoder...")

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        loss = []
        for batch_features, label_num in train_loader:
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

    torch.save(encoder.state_dict(), PATH_SAVE + 'encoder_{}_layers.pt'.format(num_layers))
    torch.save(decoder.state_dict(), PATH_SAVE + 'decoder.{}_layers.pt'.format(num_layers))
