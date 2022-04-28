from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import models

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
        self.classifier = nn.Linear(64, OUTPUTS_a)

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


def train_and_test(cnn, train_loader, test_loader, classes,
                   model_savename='savemodel.pt',
                   num_epochs = 10, batch_size = 64, learning_rate = 1E-3):

    # model_path = "/home/ubuntu/capstone/CNN/Models/Saved_Models/"

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cnn = cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    met_best = 0
    patience = 10
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

            if (i + 1) % ((len(train_loader.dataset) // batch_size)//2) == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader.dataset) // batch_size, loss.item()))

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
        f1 = f1_score(y_true, y_pred, average='micro') # micro
        print(f1_score(y_true, y_pred, average='micro'))
        print('Precision: ')
        print(precision_score(y_true, y_pred, average='micro'))
        print('Recall: ')
        print(recall_score(y_true, y_pred, average='micro'))
        print('Accuracy: ')
        print(accuracy_score(y_true, y_pred))

        patience = patience - 1

        if f1 > met_best:
            patience = 10
            met_best = f1
            torch.save(obj=cnn.state_dict(), f=model_savename)
            print('best model saved')
        if patience == 0:
            break


def evaluate_best_model(cnn,model_path, test_loader, classes):

    # model_path = '/home/ubuntu/capstone/CNN/Models/Saved_Models/'
    cnn.load_state_dict(torch.load(f=model_path))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cnn = cnn.to(device)
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
    print(f1_score(y_true, y_pred, average='micro'))
    print('Precision: ')
    print(precision_score(y_true, y_pred, average='micro'))
    print('Recall: ')
    print(recall_score(y_true, y_pred, average='micro'))
    print('Accuracy: ')
    print(accuracy_score(y_true, y_pred))

def pretrained_model(model_name,OUTPUTS_a):
    # mlb = MultiLabelBinarizer()
    # THRESHOLD = 0.5
    if model_name == 'resnet34':
    # model = models.resnet18(pretrained=True)
        model = models.resnet34(pretrained=True)
        # model = models.vgg16(pretrained=True)
        # model = models.efficientnet_b2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,OUTPUTS_a)
    # cnn = model.to(device)
    return model
def combine_and_evaluate(cnn,model_path,test_loader,xdf_dset_test,class_names):
    print('The result from the best model:')
    cnn.load_state_dict(torch.load(model_path))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cnn = cnn.to(device)
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

    result = xdf_dset_test.copy()
    result['prediction'] = [int(i) for i in y_pred]
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in class_names],
                         columns=[i for i in class_names])
    print('classification report: ')
    print(classification_report(y_true, y_pred, target_names=class_names))
    print('Confusion matrix: ')
    print(df_cm)
    f1 = f1_score(y_true, y_pred, average= 'micro')
    print(f'F1 score: {f1}')
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')

#%%
    unique_origin = result.origin.unique()
    list_pred = []
    label_list = []
    origin_list = []
    label_num_list = []
    # for i, ori in enumerate(unique_origin):
    #     subset = result[result['origin'] == ori]
    #     list_pred.append(1 if subset.prediction.mean() > 0.5 else 0)
    #     origin_list.append(ori)
    #     label_list.append(subset['label'].iloc[0])
    #     label_num_list.append(subset['label_num'].iloc[0])
#%%
   #unique_origin_list = xdf_dset_test.origin.unique()
    for i, ori in enumerate(unique_origin):
        subset = result[result['origin'] == ori]
        origin_list.append(ori)
        mode_num = subset['prediction'].mode()
        label_list.append(subset['label'].iloc[0])
        label_num_list.append(subset['label_num'].iloc[0])
        if len(mode_num) == 1:
            list_pred.append(mode_num[0])
        else:
            print('there is option 2')
            if 1 in mode_num:
                list_pred.append(1)
            elif 4 in mode_num:
                list_pred.append(4)
            elif 0 in mode_num:
                list_pred.append(0)
            elif 3 in mode_num:
                list_pred.append(3)
            else:
                list_pred.append(2)
   #
    #unique_origin['prediction'] = list_pred
#%%
    pred_result = pd.DataFrame()
    pred_result['origin'] = origin_list
    pred_result['label'] = label_list
    pred_result['label_num'] = label_num_list
    pred_result['prediction'] = list_pred
    # 0: hc(healthy control) 1: pd(parkinson disease)
    cf_matrix = confusion_matrix(pred_result['label_num'], pred_result['prediction'])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in class_names],
                         columns=[i for i in class_names])
    print('classification report: ')
    print(classification_report(pred_result['label_num'], pred_result['prediction'], target_names=class_names))
    print('Confusion matrix: ')
    print(df_cm)
    f1 = f1_score(pred_result['label_num'], pred_result['prediction'], average='micro')
    print(f'final f1 score:{f1}')
    print(f'Accuracy: {accuracy_score(pred_result.label_num, pred_result.prediction)}')
