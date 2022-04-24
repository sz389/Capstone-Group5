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

UNLABELED_BS = 26
TRAIN_BS = 12
TEST_BS = 1024

num_train_samples = 1000
num_unlabeled_samples = 2000
samples_per_class = int(num_train_samples/9) #90
unlabeled_samples_per_class = int(num_unlabeled_samples/9) #222

csv_path = "/home/ubuntu/capstone/Data/"
x = pd.read_csv(csv_path + 'mnist_train.csv')
y = x['label']
x.drop(['label'], inplace = True, axis = 1)

x_test = pd.read_csv(csv_path + 'mnist_test.csv')
y_test = x_test['label']
x_test.drop(['label'], inplace = True, axis = 1)

# unlabaled_samples_per_class = 100
# #For the train set, make sure there are equal class sizes
# x_train, x_unlabeled = x[y.values == 0].values[:unlabaled_samples_per_class], x[y.values == 0].values[samples_per_class:unlabeled_samples_per_class]
# y_train = y[y.values == 0].values[:unlabaled_samples_per_class]
#
# for i in range(1, 10):
#     x_train = np.concatenate([x_train, x[y.values == i].values[:unlabaled_samples_per_class]], axis=0)
#     y_train = np.concatenate([y_train, y[y.values == i].values[:unlabaled_samples_per_class]], axis=0)
#
#     x_unlabeled = np.concatenate([x_unlabeled, x[y.values == i].values[samples_per_class:unlabaled_samples_per_class]], axis=0)
#     unlabaled_samples_per_class = unlabaled_samples_per_class + 100


# split original train data into (20,000: 40,000)
x_train, x_unlabeled,y_train, y_unlabeled = train_test_split(x,y, train_size=1000, test_size=9000, shuffle=True, random_state=42)

#Dataloader - convert to tensors then put through dataloader
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
x_train = normalizer.fit_transform(x_train)
x_unlabeled = normalizer.transform(x_unlabeled)
x_test = normalizer.transform(x_test.values)
# x_test = x_test.values

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
            #print(predicted)
            correct += (predicted == labels.cuda()).sum()
            loss += F.nll_loss(output, labels.cuda()).item()

    return (float(correct)/len(test)) *100, (loss/len(test_loader))

net.load_state_dict(torch.load("/home/ubuntu/capstone/CNN/Models/Saved_Models/supervised_weights_random_split_1_9"))

T1 = 100
T2 = 700
af = 1

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


def semisup_train(model, train_loader, unlabeled_loader, test_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    EPOCHS = 150

    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    step = 100

    model.train()
    for epoch in tqdm(range(EPOCHS)):
        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):

            # Forward Pass to get the pseudo labels
            x_unlabeled = x_unlabeled[0].cuda()
            model.eval()
            output_unlabeled = model(x_unlabeled)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
            model.train()

            """ ONLY FOR VISUALIZATION"""
            if (batch_idx < 3) and (epoch % 10 == 0):
                unlabel.append(x_unlabeled.cpu())
                pseudo_label.append(pseudo_labeled.cpu())
            """ ********************** """

            # Now calculate the unlabeled loss using the pseudo label
            output = model(x_unlabeled)
            unlabeled_loss = alpha_weight(step) * F.nll_loss(output, pseudo_labeled)

            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()

            # For every 50 batches train one epoch on labeled data
            if batch_idx % 50 == 0:

                # Normal training procedure
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                    output = model(X_batch)
                    labeled_loss = F.nll_loss(output, y_batch)

                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()

                # Now we increment step by 1
                step += 1

        test_acc, test_loss = evaluate(model, test_loader)
        print('Epoch: {} : Alpha Weight : {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch,
                                                                                                   alpha_weight(step),
                                                                                                   test_acc, test_loss))

        """ LOGGING VALUES """
        alpha_log.append(alpha_weight(step))
        test_acc_log.append(test_acc / 100)
        test_loss_log.append(test_loss)
        """ ************** """
        model.train()


semisup_train(net, train_loader, unlabeled_loader, test_loader)
