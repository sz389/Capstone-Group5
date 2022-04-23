import torch
import sys
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evaluate_best_model, add_linear
from Models.autoencoder import cal, Encoder, Decoder, Classifier
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder
import pandas as pd
import torch as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = '/home/ubuntu/capstone/Data/'
train_df = pd.read_csv(csv_path + "emotion_train.csv")
train_df = train_df[['emotion', "Image_file_path"]]
train_df.columns=['label','id']

train_df['label'] = manual_label_encoder(train_df['label'],'emotion')


test_df = pd.read_csv(csv_path + "emotion_test.csv")
test_df = test_df[['emotion', "Image_file_path"]]
test_df.columns=['label','id']

test_df['label'] = manual_label_encoder(test_df['label'],'emotion')

train_loader = dataloader(train_df, OUTPUTS_a = 6, BATCH_SIZE = 64, IMAGE_SIZE=128)
test_loader = dataloader(test_df, OUTPUTS_a = 6, BATCH_SIZE = 64, IMAGE_SIZE=128)

batch_size = 64
epochs = 25
IMAGE_SIZE = 128
num_layers = 3
learning_rate = 0.00002
d = 64

classes = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
OUTPUTS_a = len(classes)

jj, kk = cal(IMAGE_SIZE, num_layers)
encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

PATH_SAVE = "/home/ubuntu/capstone/CNN/Models/Saved_Models/"

encoder.load_state_dict(torch.load(PATH_SAVE + "encoder_{}_layers.pt".format(num_layers)))
cnn = Classifier(encoder, d, OUTPUTS_a)

model_name = 'cnn_classifier_{}_layers.pt'.format(num_layers)
train_and_test(cnn, train_loader, test_loader, classes, model_name, epochs, batch_size, learning_rate)
evaluate_best_model(cnn, test_loader, classes, model_name)
