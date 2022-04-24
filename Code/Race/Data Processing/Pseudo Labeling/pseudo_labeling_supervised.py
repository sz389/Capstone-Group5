from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd
import sys
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evaluate_best_model, add_linear, pretrained_model, CNN9
from Utility.utility import manual_label_encoder, get_model_params
from Models.autoencoder import cal
from Utility.dataloader import dataloader, unlabeled_dataloader

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

UNLABELED_BS = 128
TRAIN_BS = 128
TEST_BS = 1024

csv_path = "/home/ubuntu/capstone/Data/"
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

batch_size = 64
d = 64
IMAGE_SIZE = 128
num_layers = 3
learning_rate = 1e-3

classes = ['Caucasian', 'African American', 'Asian']
OUTPUTS_a = len(classes)

train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
val_loader = dataloader(val_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

x_unlabeled = pd.read_csv(csv_path + "train_trim_split_spec.csv")
x_unlabeled = x_unlabeled[['native_language', 'path']]
x_unlabeled.columns=['label','id']

unlabeled_loader = unlabeled_dataloader(x_unlabeled, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

jj, kk = cal(IMAGE_SIZE, num_layers)
net = CNN(d, jj, kk, OUTPUTS_a).to('cuda')

def evaluate(model, test_loader):
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0
    y_pred = []
    y_true = []
    for images, labels in test_loader:
        images = Variable(images)
        labels = Variable(labels)
        images = images.to("cuda")
        outputs = model(images).detach().to(torch.device('cpu'))
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted)
        total += labels.size(0)
        labels = torch.argmax(labels, dim=1)
        y_true.extend(labels)
        correct += (predicted == labels).sum()
        loss += criterion(outputs, labels).item()
        f1 = f1_score(y_true, y_pred, average='weighted')

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classes],
                         columns=[i for i in classes])

    print('Confusion matrix: ')
    print(df_cm)

    #return (float(correct) / len(test_df)) * 100, (loss / len(test_loader))
    return f1, (loss / len(test_loader))

def train_supervised(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 100
    #model.train() #train
    for epoch in tqdm(range(EPOCHS)):
        correct = 0
        running_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            model.train()  # train
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            output = model(X_batch)
            labeled_loss = criterion(output, y_batch)
            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()

        if epoch % 10 == 0:
            test_acc, test_loss = evaluate(model, val_loader)
            print('Epoch: {} : Train Loss : {:.5f} | '
                  'Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch, running_loss / (10 * len(train_loader.dataset)), test_acc,
                                                                                                     test_loss))



print("Starting supervised training...")
train_supervised(net, train_loader, val_loader)

test_acc, test_loss = evaluate(net, test_loader)
print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format(test_acc, test_loss))
torch.save(net.state_dict(), '/home/ubuntu/capstone/CNN/Models/Saved_Models/supervised_weights_race')

