import torch
import sys
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evalute_best_model, add_linear
from Models.autoencoder import cal, Encoder, Decoder, Classifier
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder
import pandas as pd
import torch as nn

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

train_loader = dataloader(train_df, OUTPUTS_a = 6, BATCH_SIZE = 64, IMAGE_SIZE=128)
test_loader = dataloader(test_df, OUTPUTS_a = 6, BATCH_SIZE = 64, IMAGE_SIZE=128)

batch_size = 64
epochs = 5
IMAGE_SIZE = 128
num_layers = 3
learning_rate = 0.00002
d = 64

classes = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
#classes = list(train_df.label.unique)
OUTPUTS_a = len(classes)

jj, kk = cal(IMAGE_SIZE, num_layers)
encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

#cnn = CNN(encoded_space_dim=batch_size, jj=jj, kk=kk)
cnn = Classifier(encoder)

PATH_SAVE = "/home/ubuntu/capstone/CNN/Models/Saved_Models/"
cnn.load_state_dict(torch.load(PATH_SAVE + 'encoder_{}_layers.pt'.format(num_layers)))

#cnn = add_linear(cnn, OUTPUTS_a)
#cnn = nn.Sequential(CNN(True), nn.Linear(64, OUTPUTS_a))

train_and_test(cnn, train_loader, test_loader, classes, num_layers, epochs, batch_size, learning_rate)
evalute_best_model(cnn, test_loader, classes, num_layers)