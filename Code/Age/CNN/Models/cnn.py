from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision import models
from tqdm import tqdm
import sys
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.autoencoder import cal

class CNN(nn.Module):
    def __init__(self, encoded_space_dim, jj, kk, OUTPUTS_a):
        super().__init__()
        channels = 3 ; ch1 = 16 ; ch2 = 32 ; ch3 = 64
        kernel_size = (4, 4); padding = (0, 0) ; stride = (2, 2)

        self.enc1 = nn.Conv2d(in_channels= channels, out_channels=ch1, kernel_size=kernel_size,  stride=stride, padding=padding)
        #self.relu = nn.ReLU(True)
        self.enc2= nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=kernel_size,  stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(ch2)
        #self.relu = nn.ReLU(True)
        self.enc3= nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=kernel_size,  stride=stride, padding=padding)
        #self.relu = nn.ReLU(True)
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(nn.Linear(int(ch3 * jj * kk), 128),  nn.Linear(128, encoded_space_dim)) # nn.ReLU(True)
        self.classifier = nn.Linear(encoded_space_dim, OUTPUTS_a)

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

def train_and_test(cnn, train_loader, val_loader, classes, model_name, model_path,
                   num_epochs = 10,
                   batch_size = 64,
                   learning_rate = 1E-3
                   ):


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cnn = cnn.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    #patience = 15
    met_best = 0

    for epoch in tqdm(range(num_epochs)):

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

            if (i + 1) % ((len(train_loader.dataset) // batch_size)//2) == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader.dataset) // batch_size, loss.item()))
        # Test the Model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        for images, labels in tqdm(val_loader):
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

        #patience = patience - 1

        if f1 > met_best:
            #patience = 15
            met_best = f1
            torch.save(obj=cnn.state_dict(), f=model_path + f"{model_name}")

        # if patience == 0:
        #     break


def evaluate_best_model(cnn, test_loader, classes, model_name,
                       model_path):

    cnn.load_state_dict(torch.load(f=model_path + model_name))
    cnn.eval().to('cuda')  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    for images, labels in tqdm(test_loader):
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

class add_linear(nn.Module):
    def __init__(self, CNN, model_output, OUTPUTS_a):
        super().__init__()
        self.model = CNN
        self.classifier = nn.Linear(model_output, OUTPUTS_a)

    def forward(self,x):
        x = self.model(x)
        x = self.classifier(x)
        return x

class CNN9(nn.Module):
    def __init__(self, OUTPUTS_a):
        super(CNN9, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.p = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32,64,(3,3))
        self.convnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64,128,(3,3))
        self.convnorm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, (3, 3))
        self.convnorm5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, (3, 3))
        self.convnorm6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 256, (3, 3))
        self.convnorm7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 256, (3, 3))
        self.convnorm8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, (3, 3))
        self.convnorm9 = nn.BatchNorm2d(256)

        # self.conv10 = nn.Conv2d(256, 256, (3, 3))
        # self.convnorm10 = nn.BatchNorm2d(256)
        #
        # self.conv11 = nn.Conv2d(256, 256, (3, 3))
        # self.convnorm11 = nn.BatchNorm2d(256)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(256, OUTPUTS_a)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.p(self.pad1(self.convnorm2(self.act(self.conv2(x)))))
        x = self.pad1(self.convnorm3(self.act(self.conv3(x))))
        x = self.p(self.pad1(self.convnorm4(self.act(self.conv4(x)))))
        x = self.pad1(self.convnorm5(self.act(self.conv5(x))))
        x = self.p(self.pad1(self.convnorm6(self.act(self.conv6(x)))))
        x = self.pad1(self.convnorm7(self.act(self.conv7(x))))
        x = self.p(self.pad1(self.convnorm8(self.act(self.conv8(x)))))
        x = self.act(self.convnorm9(self.act(self.conv9(x))))
        # x = self.p(self.pad1(self.convnorm10(self.act(self.conv10(x)))))
        # x = self.act(self.convnorm11(self.act(self.conv11(x))))

        return self.linear(self.global_avg_pool(x).view(-1, 256))

def pretrained_model(model_name,OUTPUTS_a):
    if model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,OUTPUTS_a)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, OUTPUTS_a)
    elif model_name == "cnn9":
        model = CNN9(OUTPUTS_a)
    elif model_name == "cnn3":
        d = 64; IMAGE_SIZE = 128; num_layers = 3
        jj, kk = cal(IMAGE_SIZE, num_layers)
        model = CNN(d, jj, kk, OUTPUTS_a)

    return model

