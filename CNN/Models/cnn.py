from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

class CNN(nn.Module):
    def __init__(self, encoded_space_dim, jj, kk, OUTPUTS_a):
        super().__init__()
        channels = 3 ; ch1 = 16 ; ch2 = 32 ; ch3 = 64
        kernel_size = (4, 4); padding = (0, 0) ; stride = (2, 2)

        self.enc1 = nn.Conv2d(in_channels=channels, out_channels=ch1, kernel_size=kernel_size,  stride=stride, padding=padding)
        #self.relu = nn.ReLU(True)
        self.enc2= nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=kernel_size,  stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(ch2)
        #self.relu = nn.ReLU(True)
        self.enc3= nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=kernel_size,  stride=stride, padding=padding)
        #self.relu = nn.ReLU(True)
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(nn.Linear(int(ch3 * jj * kk), 128),  nn.Linear(128, encoded_space_dim)) # nn.ReLU(True)
        self.classifier = nn.linear(64, OUTPUTS_a)

    def forward(self, x):
        # input: [64, 3, 128, 128]
        x = self.enc1(x)  # output: [64, 16, 63, 63]
        # note: with padding (0,0), height - 1 and width - 1, with padding (1,1) height = 64, width = 64
        # x = self.relu(x)
        x = torch.tanh(x)  # input/output: [64, 16, 63, 63]
        x = self.enc2(x)  # output: [64, 32, 30, 30]
        x = self.batchnorm(x)  # output: [64, 32, 30, 30]
        # x = self.relu(x)
        x = torch.tanh(x)  # output: [64, 32, 30, 30]
        x = self.enc3(x)  # output: [64, 64, 14, 14]
        x = torch.tanh(x)  # output: [64, 64, 14, 14]
        y = self.flatten(x) #output: [64, 12544] = [64, 64 * 14 * 14]
        x = self.encoder_lin(y) #output: [64, 64]
        x = self.classifier(x)
        return x #return: [64, 64]


def train_and_test(cnn, train_loader, test_loader, classes, num_layers, num_epochs = 10, batch_size = 64, learning_rate = 1E-3):

    model_path = "/home/ubuntu/capstone/CNN/Models/Saved_Models/"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):

            cnn.train()

            images = Variable(images)
            labels = Variable(labels)
            images = images.to('cuda')
            labels = labels.to("cuda")

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader.df) // batch_size, loss.item()))

        # Test the Model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        for images, labels in test_loader:
            images = Variable(images).to("cuda")
            outputs = cnn(images).detach().to(torch.device('cpu'))
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted)
            # print(predicted)
            total += labels.size(0)
            labels = torch.argmax(labels, dim=1)
            y_true.extend(labels)
            correct += (predicted == labels).sum()

        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classes],
                             columns=[i for i in classes])

        print('Confusion matrix: ')
        print(df_cm)
        print('F1 score: ')
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f1_score(y_true, y_pred, average='weighted'))
        print('Precision: ')
        print(precision_score(y_true, y_pred, average='weighted'))
        print('Recall: ')
        print(recall_score(y_true, y_pred, average='weighted'))
        print('Accuracy: ')
        print(accuracy_score(y_true, y_pred))

        patience = patience - 1

        if f1 > met_best:
            patience = 4
            met_best = f1
            torch.save(obj=cnn.state_dict(), f=model_path + 'cnn_{}_layers.pt'.format(num_layers))

        if patience == 0:
            break


def evalute_best_model(cnn, test_loader, classes, num_layers):

    model_path = '/home/ubuntu/capstone/CNN/Models/Saved_Models/'
    cnn.load_state_dict(torch.load(f=model_path + 'cnn_{}_layers.pt'.format(num_layers)))
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    for images, labels in test_loader:
        images = Variable(images).to("cuda")
        outputs = cnn(images).detach().to(torch.device('cpu'))
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted)
        # print(predicted)
        total += labels.size(0)
        labels = torch.argmax(labels, dim=1)
        y_true.extend(labels)
        correct += (predicted == labels).sum()

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classes],
                         columns=[i for i in classes])
    print()
    print("Final Scores")
    print()
    print('Confusion matrix: ')
    print(df_cm)
    print('F1 score: ')
    print(f1_score(y_true, y_pred, average='weighted'))
    print('Precision: ')
    print(precision_score(y_true, y_pred, average='weighted'))
    print('Recall: ')
    print(recall_score(y_true, y_pred, average='weighted'))
    print('Accuracy: ')
    print(accuracy_score(y_true, y_pred))
