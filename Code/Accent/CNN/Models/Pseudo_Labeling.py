#%%
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torch.autograd import Variable
import pandas as pd
#sys.path.insert(1, '/home/ubuntu/Capstone/CNN')
from Models.cnn import pretrained_model #train_and_test, evaluate_best_model, add_linear, pretrained_model, CNN9
from pseudo_labeling_utility import manual_label_encoder, dataloader, unlabeled_dataloader
import argparse

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    f1 = f1_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classes],
                         columns=[i for i in classes])

    print('Confusion matrix: ')
    print(df_cm)

    #return (float(correct)/len(test_df)) * 100, (loss/len(test_loader))
    return acc* 100, f1 * 100, (loss/len(test_loader))




def alpha_weight(epoch): #150 epochs, 9 steps per epoch, after 150 epochs, step = 150 * 9
    T1 = 100
    T2 = 700  # change with epoch value
    af = 0.50  # correlates to confidence level in unlabeled data

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
    met_best = 0
    patience = 10
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
            if batch_idx % 30 == 0: #9 times every epoch

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

        val_acc, val_f1, val_loss = evaluate(model, val_loader)
        print('Epoch: {} : Alpha Weight : {:.5f} | Val Acc : {:.5f} | Val Loss : {:.3f} '.format(epoch,
                                                                                                   alpha_weight(step),
                                                                                                   val_acc, val_f1,val_loss))
        patience = patience - 1
        if val_acc > met_best:
            #patience = 15
            met_best = val_f1
            torch.save(obj=net.state_dict(), f= model_path + "/new_semisupervised_weights_accent")
        if patience == 0:
            break
        model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=None, type=str, required=True)  # Path of csv to load
    parser.add_argument("--train_csv", default=None, type=str, required=True)  # Path to save the csv file
    parser.add_argument("--test_csv", default=None, type=str, required=True)
    parser.add_argument("--val_csv", default=None, type=str, required=True)
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument("--unlabel_csv_path", default=None, type=str, required=True)
    parser.add_argument("--unlabel_csv_name", default=None, type=str, required=True)
    args = parser.parse_args()

    csv_path = args.csv_path
    train_data = args.train_csv
    test_data = args.test_csv
    val_data = args.val_csv
    model_path = args.model_path
    csv_path1 = args.unlabel_csv_path
    x_unlabeled_data = args.unlabel_csv_name

   # csv_path = "/home/ubuntu/Capstone/data_train_val_test/"
    train_df = pd.read_csv(csv_path + train_data)
    # %%
    train_df = train_df[['native_language', "path"]]
    train_df.columns = ['label', 'id']

    train_df['label'] = manual_label_encoder(train_df['label'], 'accent')
    # %%
    val_df = pd.read_csv(csv_path + val_data)
    val_df = val_df[['native_language', "path"]]
    val_df.columns = ['label', 'id']
    val_df['label'] = manual_label_encoder(val_df['label'], 'accent')
    # %%
    test_df = pd.read_csv(csv_path + test_data)
    test_df = test_df[['native_language', "path"]]
    test_df.columns = ['label', 'id']

    test_df['label'] = manual_label_encoder(test_df['label'], 'accent')

    batch_size = 64
    d = 64
    IMAGE_SIZE = 128
    num_layers = 3
    learning_rate = 1e-3

    classes = ['arabic', 'english', 'french', 'mandarin', 'spanish']
    OUTPUTS_a = len(classes)

    train_loader = dataloader(train_df, OUTPUTS_a=OUTPUTS_a, BATCH_SIZE=batch_size, IMAGE_SIZE=IMAGE_SIZE)
    val_loader = dataloader(val_df, OUTPUTS_a=OUTPUTS_a, BATCH_SIZE=batch_size, IMAGE_SIZE=IMAGE_SIZE)
    test_loader = dataloader(test_df, OUTPUTS_a=OUTPUTS_a, BATCH_SIZE=batch_size, IMAGE_SIZE=IMAGE_SIZE)


    # %%
   # csv_path1 = '/home/ubuntu/Capstone/race_data/CREMA-D/'
    x_unlabeled = pd.read_csv(csv_path1 + x_unlabeled_data)  #"Mel_Spectrograms.csv"
    # %%
    x_unlabeled = x_unlabeled[['img_name']]
    x_unlabeled['path'] = csv_path1 + '/mel_spectrograms/' + x_unlabeled['img_name']
    x_unlabeled.columns = ['label', 'id']  # the label here is the image name

    unlabeled_loader = unlabeled_dataloader(x_unlabeled, BATCH_SIZE=batch_size, IMAGE_SIZE=IMAGE_SIZE)
    net = pretrained_model('resnet34', OUTPUTS_a).to('cuda')
    net.load_state_dict(torch.load(model_path + "/resnet34.pt"))
    semisup_train(net, train_loader, unlabeled_loader, val_loader)

    net.load_state_dict(torch.load(f= model_path + "/new_semisupervised_weights_accent"))
    test_acc,test_f1, test_loss = evaluate(net, test_loader)
    print('Test Acc : {:.5f} | Test_F1: {:.3f} | Test Loss : {:.3f} '.format(test_acc, test_f1,test_loss))




