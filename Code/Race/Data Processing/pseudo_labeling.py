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
import argparse
from tqdm import tqdm
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evaluate_best_model, add_linear, pretrained_model, CNN9
from Utility.utility import manual_label_encoder, get_model_params, get_classes
from Models.autoencoder import cal
from Utility.dataloader import dataloader, unlabeled_dataloader

# Concept from : https://github.com/peimengsui/semi_supervised_mnist

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
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 2, index=[i for i in classes],
                         columns=[i for i in classes])
    print()
    print('Confusion matrix: ')
    print(df_cm)

    #return (float(correct)/len(test_df)) * 100, (loss/len(test_loader))
    return acc * 100, f1 * 100, (loss/len(test_loader))


T1 = 100
T2 = 700 # change with epoch value
af = 0.50 # correlates to confidence level in unlabeled data

def alpha_weight(epoch): #150 epochs, 9 steps per epoch, after 150 epochs, step = 150 * 9
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
        return ((epoch - T1) / (T2 - T1)) * af

acc_scores = []
unlabel = []
pseudo_label = []

alpha_log = []
test_acc_log = []
test_loss_log = []

def semisup_train(model, train_loader, unlabeled_loader, val_loader, model_path, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 150
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
            if batch_idx % 25 == 0: #4 times every epoch

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
        print('Epoch: {} : Alpha Weight : {:.5f} | Val Acc: {:.5f} | Val F1 : {:.5f} | Val Loss : {:.3f} '.format(epoch,
                                                                                                   alpha_weight(step),
                                                                                                   val_acc, val_f1, val_loss))
        patience = patience - 1
        if val_acc > met_best:
            patience = 20
            met_best = val_f1
            torch.save(obj=net.state_dict(), f=model_path + model_name)

        model.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    parser.add_argument('--category', default=None, type=str, required=True) #category (Ex. emotion, race, etc.)
    parser.add_argument("--model",default=None, type=str, required=True)
    parser.add_argument("--cnn_param_file", default=None, type=str, required=True)
    parser.add_argument("--pseudolabeling_param_file", default=None, type=str, required=True)
    parser.add_argument("--model_save_and_load_path", default=None, type=str, required=True)
    parser.add_argument("--unlabeled_csv", default=None, type=str, required=True)
    parser.add_argument("--train_csv", default=f"race_train.csv", type=str, required=False)  # train_csv
    parser.add_argument("--val_csv", default=f"race_val.csv", type=str, required=False)  # val_csv
    parser.add_argument("--test_csv", default=f"race_test.csv", type=str, required=False)  # test_csv
    parser.add_argument("--epochs", default=150, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--learning_rate", default=1e-3, type=int, required=False)

    args = parser.parse_args()
    csv_load_path = args.csv_load_path
    category = args.category
    model = args.model
    cnn_param_file = args.cnn_param_file
    model_save_and_load_path = args.model_save_and_load_path
    pseudolabeling_param_file = args.pseudolabeling_param_file
    unlabeled_csv = args.unlabeled_csv
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv

    train_df = pd.read_csv(csv_load_path + train_csv)
    train_df = train_df[[category, "Image_file_path"]]
    train_df.columns = ['label', 'id']

    train_df['label'] = manual_label_encoder(train_df['label'], category)

    val_df = pd.read_csv(csv_load_path + val_csv)
    val_df = val_df[[category, "Image_file_path"]]
    val_df.columns = ['label', 'id']

    val_df['label'] = manual_label_encoder(val_df['label'], category)

    test_df = pd.read_csv(csv_load_path + train_csv)
    test_df = test_df[[category, "Image_file_path"]]
    test_df.columns = ['label', 'id']

    test_df['label'] = manual_label_encoder(test_df['label'], category)

    d = 64
    IMAGE_SIZE = 128
    num_layers = 3

    classes = get_classes(category)
    OUTPUTS_a = len(classes)

    train_loader = dataloader(train_df, OUTPUTS_a=OUTPUTS_a, BATCH_SIZE=batch_size, IMAGE_SIZE=IMAGE_SIZE)
    val_loader = dataloader(val_df, OUTPUTS_a=OUTPUTS_a, BATCH_SIZE=batch_size, IMAGE_SIZE=IMAGE_SIZE)
    test_loader = dataloader(test_df, OUTPUTS_a=OUTPUTS_a, BATCH_SIZE=batch_size, IMAGE_SIZE=IMAGE_SIZE)

    x_unlabeled = pd.read_csv(csv_load_path + unlabeled_csv)
    x_unlabeled = x_unlabeled[['native_language', 'path']] #assuming unlabeled data is accent data
    x_unlabeled.columns = ['label', 'id']

    unlabeled_loader = unlabeled_dataloader(x_unlabeled, BATCH_SIZE=batch_size, IMAGE_SIZE=IMAGE_SIZE)

    net = pretrained_model(model, OUTPUTS_a).to('cuda')
    net.load_state_dict(torch.load(model_save_and_load_path + cnn_param_file))

    print("Starting Semi-Supervised Learning...")

    semisup_train(net, train_loader, unlabeled_loader, val_loader, model_save_and_load_path, pseudolabeling_param_file)

    net.load_state_dict(torch.load(f=model_save_and_load_path + pseudolabeling_param_file))
    print()
    print("Evaluating...")
    print()
    test_acc, test_f1, test_loss = evaluate(net, test_loader)
    print('Test Acc : {:.5f} | Test F1 : {:.5f} | Test Loss : {:.3f} '.format(test_acc, test_f1, test_loss))
