import torch
import torch.nn as nn
import sys
import pandas as pd
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evaluate_best_model, add_linear, pretrained_model, CNN9
from Models.autoencoder import cal
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder, get_classes
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    parser.add_argument('--category', default=None, type=str, required=True) #category (Ex. emotion, race, etc.)
    parser.add_argument("--model",default=None, type=str, required=True)
    parser.add_argument("--model_save_path", default=None, type=str, required=True)
    parser.add_argument("--epochs", default=30, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--learning_rate", default=1e-3, type=int, required=False)
    parser.add_argument("--train_csv", default=f"emotion_train.csv", type=str, required=False)  # train_csv
    parser.add_argument("--val_csv", default=f"emotion_val.csv", type=str, required=False)  # val_csv
    parser.add_argument("--test_csv", default=f"emotion_test.csv", type=str, required=False)  # test_csv

    args = parser.parse_args()
    csv_load_path = args.csv_load_path
    category = args.category
    model = args.model
    model_save_path = args.model_save_path
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv

    train_df = pd.read_csv(csv_load_path + train_csv)
    train_df = train_df[[category, "Image_file_path"]]
    train_df.columns=['label','id']
    train_df['label'] = manual_label_encoder(train_df['label'],category)

    val_df = pd.read_csv(csv_load_path + val_csv)
    val_df = val_df[[category, "Image_file_path"]]
    val_df.columns=['label','id']
    val_df['label'] = manual_label_encoder(val_df['label'],category)

    test_df = pd.read_csv(csv_load_path + test_csv)
    test_df = test_df[[category, "Image_file_path"]]
    test_df.columns=['label','id']
    test_df['label'] = manual_label_encoder(test_df['label'],category)

    classes = get_classes(category)
    OUTPUTS_a = len(classes)

    d = 64
    IMAGE_SIZE = 128

    train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
    val_loader = dataloader(val_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
    test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

    cnn = pretrained_model(model, OUTPUTS_a).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    model_name = f'{model}_{epochs}_{batch_size}_{learning_rate}.pt'

    print("Training...")
    train_and_test(cnn, train_loader, val_loader, classes, model_name, model_save_path, epochs, batch_size, learning_rate)
    print()
    print("Evaluating...")
    print()
    evaluate_best_model(cnn, test_loader, classes, model_name, model_save_path)