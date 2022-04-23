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

#emotion
csv_path = '/home/ubuntu/capstone/Data/'
# train_df = pd.read_csv(csv_path + "emotion_train.csv")
# train_df = train_df[['emotion', "Image_file_path"]]
# train_df.columns=['label','id']
#
# train_df['label'] = manual_label_encoder(train_df['label'],'emotion')
#
# val_df = pd.read_csv(csv_path + "emotion_val.csv")
# val_df = val_df[['emotion', "Image_file_path"]]
# val_df.columns=['label','id']
#
# val_df['label'] = manual_label_encoder(val_df['label'],'emotion')
#
# test_df = pd.read_csv(csv_path + "emotion_test.csv")
# test_df = test_df[['emotion', "Image_file_path"]]
# test_df.columns=['label','id']
#
# test_df['label'] = manual_label_encoder(test_df['label'],'emotion')
#
# classes = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
train_df = pd.read_csv(csv_path + "race_train.csv")
train_df = train_df[['Race', "Image_file_path"]]
train_df.columns=['label','id']

train_df['label'] = manual_label_encoder(train_df['label'],'race')

val_df = pd.read_csv(csv_path + "race_val.csv")
val_df = val_df[['Race', "Image_file_path"]]
val_df.columns=['label','id']

val_df['label'] = manual_label_encoder(val_df['label'],'race')

test_df = pd.read_csv(csv_path + "race_test.csv")
test_df = test_df[['Race', "Image_file_path"]]
test_df.columns=['label','id']

test_df['label'] = manual_label_encoder(test_df['label'],'race')

classes = ['Caucasian', 'African American', 'Asian']
OUTPUTS_a = len(classes)

batch_size = 64
epochs = 60
IMAGE_SIZE = 128
num_layers = 3
learning_rate = 1e-3
d = 64

train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
val_loader = dataloader(val_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

jj, kk = cal(IMAGE_SIZE, num_layers)
encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)
#decoder = Decoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

PATH_SAVE = "/home/ubuntu/capstone/CNN/Models/Saved_Models/"

encoder.load_state_dict(torch.load(PATH_SAVE + "encoder_{}_layers.pt".format(num_layers)))
cnn = Classifier(encoder, d, OUTPUTS_a)

model_name = 'cnn_classifier_{}_layers_race_with_val_with_AE.pt'.format(num_layers)

train_and_test(cnn,
               train_loader,
               val_loader,
               classes,
               model_name,
               epochs,
               batch_size,
               learning_rate)

evaluate_best_model(cnn,
                    test_loader,
                    classes,
                    model_name)
