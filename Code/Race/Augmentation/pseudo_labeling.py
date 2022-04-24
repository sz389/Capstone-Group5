#matplotlib inline
#from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Read in labeled data, split into audio and label, split into train and test
import pandas as pd

UNLABELED_BS = 8
TRAIN_BS = 32
TEST_BS = 1024

csv_path = "/home/ubuntu/capstone/Data/"
x = pd.read_csv(csv_path + 'mnist_train.csv')
y = x['label']
x.drop(['label'], inplace = True, axis = 1)

x_test = pd.read_csv(csv_path + 'mnist_test.csv')
y_test = x_test['label']
x_test.drop(['label'], inplace = True, axis = 1)

# unlabeled_samples_per_class = 100
# samples_per_class = 90
#
# #For the train set, make sure there are equal class sizes
# x_train, x_unlabeled = x[y.values == 0].values[:samples_per_class], x[y.values == 0].values[samples_per_class:unlabeled_samples_per_class]
# y_train = y[y.values == 0].values[:unlabeled_samples_per_class]
#
# for i in range(1, 10):
#     x_train = np.concatenate([x_train, x[y.values == i].values[:labeled_samples_per_class]], axis=0)
#     y_train = np.concatenate([y_train, y[y.values == i].values[:labeled_samples_per_class]], axis=0)
#
#     x_unlabeled = np.concatenate([x_unlabeled, x[y.values == i].values[samples_per_class:unlabeled_samples_per_class]], axis=0)
#     unlabeled_samples_per_class = unlabeled_samples_per_class + 100
#     labeled_samples_per_class = labeled_samples_per_class + 100

x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x,y, train_size= 500, test_size=9000, shuffle=True, random_state=42, stratify=y)


#Dataloader - convert to tensors then put through dataloader
from sklearn.preprocessing import Normalizer

print("Length of Labeled Data: ", len(x_train))
print("Length of Unlabeled Data: ", len(x_train))
normalizer = Normalizer()
x_train = normalizer.fit_transform(x_train)
x_unlabeled = normalizer.transform(x_unlabeled)
x_test = normalizer.transform(x_test.values)

x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = pd.Series.to_numpy(y_train)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test.values).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = TRAIN_BS, shuffle = True, num_workers = 8)

unlabeled_train = torch.from_numpy(x_unlabeled).type(torch.FloatTensor)

unlabeled = torch.utils.data.TensorDataset(unlabeled_train)
unlabeled_loader = torch.utils.data.DataLoader(unlabeled, batch_size = UNLABELED_BS, shuffle = True, num_workers = 8)

test_loader = torch.utils.data.DataLoader(test, batch_size = TEST_BS, shuffle = True, num_workers = 8)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(640, 150)
        self.fc2 = nn.Linear(150, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.log_softmax(x)
        return x

net = Net().cuda()

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda()
            output = model(data)
            predicted = torch.max(output,1)[1]
            correct += (predicted == labels.cuda()).sum()
            loss += F.nll_loss(output, labels.cuda()).item()

    return (float(correct)/len(test)) *100, (loss/len(test_loader))


from tqdm import tqdm


def train_supervised(model, train_loader, test_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    EPOCHS = 100
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        correct = 0
        running_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            output = model(X_batch)
            labeled_loss = F.nll_loss(output, y_batch)

            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()

        if epoch % 10 == 0:
            test_acc, test_loss = evaluate(model, test_loader)
            print('Epoch: {} : Train Loss : {:.5f} | '
                  'Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch, running_loss / (10 * len(train)), test_acc,
                                                                                                     test_loss))
            model.train()

print("Starting supervised training...")
train_supervised(net, train_loader, test_loader)

test_acc, test_loss = evaluate(net, test_loader)
print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format(test_acc, test_loss))
torch.save(net.state_dict(), '/home/ubuntu/capstone/CNN/Models/Saved_Models/supervised_weights_random_split_1_9')

