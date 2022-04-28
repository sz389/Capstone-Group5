
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from torch.autograd import Variable
import pandas as pd
from CNN.models.cnn import CNN, train_and_test, evaluate_best_model, pretrained_model, CNN9,define_cnn, combine_and_evaluate
import os
import cv2
from torch.utils import data
from utility import dataloader,get_n_params

torch.manual_seed(42)
np.random.seed(42)

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''

    def __init__(self, df, OUTPUTS_a, transform=None, IMAGE_SIZE=128):
        self.df = df
        self.transform = transform
        self.IMAGE_SIZE = IMAGE_SIZE
        self.OUTPUTS_a = OUTPUTS_a

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_name = self.df.id.iloc[index]
        img = cv2.imread(file_name)
        img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        X = torch.FloatTensor(img)
        X = torch.reshape(X, (3, self.IMAGE_SIZE, self.IMAGE_SIZE))/255
        return X

def unlabel_dataloader(xdf_dset, OUTPUTS_a, BATCH_SIZE = 64, IMAGE_SIZE=128,shuffle=True):

    params = {'batch_size': BATCH_SIZE,
              'shuffle':shuffle}
    training_set = Dataset(xdf_dset, OUTPUTS_a,IMAGE_SIZE= IMAGE_SIZE, transform=None)
    training_generator = data.DataLoader(training_set, **params)

    return training_generator


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
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classnames],
                         columns=[i for i in classnames])

    print('Confusion matrix: ')
    print(df_cm)

    #return (float(correct)/len(test_df)) * 100, (loss/len(test_loader))
    return f1 * 100, (loss/len(test_loader))



def alpha_weight(epoch): #150 epochs, 9 steps per epoch, after 150 epochs, step = 150 * 9
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
        return ((epoch - T1) / (T2 - T1)) * af

# Concept from : https://github.com/peimengsui/semi_supervised_mnist




def semisup_train(model, train_loader, unlabeled_loader, val_loader,epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = epochs
    met_best = 0
    patience = 20
    # Instead of using current epoch we use a "step" variable to calculate alpha_weight
    # This helps the model converge faster
    step = 100
    # model.eval()
    for epoch in tqdm(range(EPOCHS)):
        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):

            # Forward Pass to get the pseudo labels
            x_unlabeled = Variable(x_unlabeled).to('cuda')
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

            # For every 10 batches train one epoch on labeled data
            if batch_idx % 10 == 0: #4 times every epoch

                # Normal training procedure
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch, y_batch = Variable(X_batch).to("cuda"), Variable(y_batch).to("cuda")
                    output = model(X_batch)
                    labeled_loss = criterion(output, y_batch)

                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()

                # Now we increment step by 1
                step += 1

        val_acc, val_loss = evaluate(model, val_loader)
        print('Epoch: {} : Alpha Weight : {:.5f} | Val Acc : {:.5f} | Val Loss : {:.3f} '.format(epoch,
                                                                                                   alpha_weight(step),
                                                                                                   val_acc, val_loss))
        patience = patience - 1
        if val_acc > met_best:
            patience = 10
            met_best = val_acc
            torch.save(obj=net.state_dict(), f=model_dir+"supervised_weights_disease_ifbatch10.pt")
        if val_acc ==100:
            break
        if patience==0:
            break
        model.train()
from tqdm import tqdm
import argparse
if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    # make sure the unlabeled csv has a column named "id" which is the path to the images (mel-spectrograms)
    parser.add_argument("--unlabeled_csv",default=None,type = str, required=True)

    parser.add_argument("--epochs", default=150, type=int, required=False)
    parser.add_argument("--model_dir", default=None, type=str, required=True)

    args = parser.parse_args()

    epochs = args.epochs
    unlabeled_path = args.unlabeled_csv

    # train_csv = args.train_csv
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    csv_load_path = args.csv_load_path


    acc_scores = []
    unlabel = []
    pseudo_label = []

    alpha_log = []
    test_acc_log = []
    test_loss_log = []
    batch_size = 64
    d = 64
    IMAGE_SIZE = 128
    num_layers = 3
    learning_rate = 1e-3
    df_unlabel = pd.read_csv(unlabeled_path)
    # df_unlabel = pd.read_csv("/home/ubuntu/Capstone/LJSpeech-1.1/LJimg.csv")
    df_train = pd.read_csv(csv_load_path+"/KCL_train_trim_split_spec.csv")
    df_val = pd.read_csv(csv_load_path+"/KCL_valid_trim_split_spec.csv")
    df_test = pd.read_csv(csv_load_path+"/KCL_test_trim_split_spec.csv")
    # model_dir = "/home/ubuntu/Capstone/saved_model/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    classnames = ['hc','pd']
    OUTPUTS_a = len(classnames)
    batch_size = 64
    num_epochs = 50
    train_loader = dataloader(xdf_dset=df_train,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=True)
    val_loader = dataloader(xdf_dset=df_val,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=True)
    test_loader = dataloader(xdf_dset=df_test,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=False)
    unlabeled_loader = unlabel_dataloader(df_unlabel,OUTPUTS_a=OUTPUTS_a,
                              BATCH_SIZE = batch_size, IMAGE_SIZE=128,shuffle=False)
    net = define_cnn(OUTPUTS_a=len(classnames))
    net.to('cuda')

    model_savename = model_dir + '/cnn9_parkinson.pt'
    net.load_state_dict(torch.load(model_savename))

    T1 = 100
    T2 = 700 # change with epoch value
    af = 0.50 # correlates to confidence level in unlabeled data
    OUTPUTS_a = len(classnames)
    semisup_train(net, train_loader, unlabeled_loader, val_loader,epochs)

    net.load_state_dict(torch.load(f=model_dir+"/supervised_weights_disease_ifbatch10.pt"))
    test_acc, test_loss = evaluate(net, test_loader)
    print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format(test_acc, test_loss))
