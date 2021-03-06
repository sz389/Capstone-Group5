import torch
import sys
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evaluate_best_model, add_linear
from Models.autoencoder import cal, Encoder, Decoder
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder, get_classes
import pandas as pd
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    parser.add_argument('--category', default=None, type=str, required=True) #category (Ex. emotion, race, etc.)
    parser.add_argument("--model_load_path", default=None, type=str, required=True)
    parser.add_argument("--train_csv", default=f"age_train.csv", type=str, required=False)  # train_csv
    parser.add_argument("--epochs", default=5, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--learning_rate", default=1e-3, type=int, required=False)

    args = parser.parse_args()
    csv_load_path = args.csv_load_path
    category = args.category
    PATH_SAVE = args.model_load_path
    train_csv = args.train_csv
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    train_df = pd.read_csv(csv_load_path + train_csv)
    train_df = train_df[[category, "Image_file_path"]]
    train_df.columns = ['label', 'id']

    train_df['label'] = manual_label_encoder(train_df['label'], category)

    classes = get_classes(category)
    OUTPUTS_a = len(classes)

    d = 64
    IMAGE_SIZE = 128
    num_layers = 3  # change based on autoencoder architecture

    train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

    jj, kk = cal(IMAGE_SIZE, num_layers)
    encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
    decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

    encoder.load_state_dict(torch.load(PATH_SAVE + 'encoder_{}_layers.pt'.format(num_layers)))
    decoder.load_state_dict(torch.load(PATH_SAVE + 'decoder.{}_layers.pt'.format(num_layers)))

    print("Starting Autoencoder...")

    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        encoder.eval()
        decoder.eval()
        loss = []
        for batch_features, label in train_loader:
            batch_features = batch_features.to(device)
            encoded_data = encoder(batch_features.float())
            decoded_data = decoder(encoded_data)
            train_loss = criterion(decoded_data, batch_features.float())
            train_loss.backward()
            loss.append(train_loss.detach().cpu().numpy().item())
        losses = np.mean(loss)

        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, losses))

