from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torch.autograd import Variable
import pandas as pd
import sys
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

    #return (float(correct)/len(test_df)) * 100, (loss/len(test_loader))
    return f1 * 100, (loss/len(test_loader))

net.load_state_dict(torch.load("/home/ubuntu/capstone/CNN/Models/Saved_Models/cnn_3_layers_race_with_val.pt"))

T1 = 100
T2 = 700
af = 0.5

def alpha_weight(epoch):
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
        return ((epoch - T1) / (T2 - T1)) * af

# Concept from : https://github.com/peimengsui/semi_supervised_mnist

from tqdm import tqdm

acc_scores = []
unlabel = []
pseudo_label = []

alpha_log = []
test_acc_log = []
test_loss_log = []


def semisup_train(model, train_loader, unlabeled_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 150

    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    step = 100
    model.eval()
    for epoch in tqdm(range(EPOCHS)):
        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):

            # Forward Pass to get the pseudo labels
            x_unlabeled = x_unlabeled.to('cuda')
            model.eval() #evaluate model that was loaded in
            output_unlabeled = model(x_unlabeled)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
            model.train() #train model after inputting first unlabeled data

            # Now calculate the unlabeled loss using the pseudo label
            output = model(x_unlabeled)
            unlabeled_loss = alpha_weight(step) * criterion(output, pseudo_labeled)

            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()

            # For every 50 batches train one epoch on labeled data
            if batch_idx % 50 == 0:

                # Normal training procedure
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch = Variable(X_batch)
                    y_batch = Variable(y_batch)
                    X_batch = X_batch.to('cuda')
                    y_batch = y_batch.to('cuda')
                    #model.train()
                    optimizer.zero_grad()
                    output = model(X_batch)
                    #predicted = torch.max(output, 1)[1]
                    #y_batch_pred = torch.max(y_batch, 1)[1]
                    #labeled_loss = F.nll_loss(predicted, y_batch_pred)
                    labeled_loss = criterion(output, y_batch)
                    labeled_loss.backward()
                    optimizer.step()

                # Now we increment step by 1
                step += 1


        val_acc, val_loss = evaluate(model, val_loader)
        print('Epoch: {} : Alpha Weight : {:.5f} | Val Acc : {:.5f} | Val Loss : {:.3f} '.format(epoch,
                                                                                                   alpha_weight(step),
                                                                                                   val_acc, val_loss))
        model.train()


semisup_train(net, train_loader, unlabeled_loader, val_loader)

test_acc, test_loss = evaluate(net, test_loader)
print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format(test_acc, test_loss))

