import torch
import sys
sys.path.insert(1, '/home/ubuntu/capstone/CNN')
from Models.cnn import CNN, train_and_test, evaluate_best_model, add_linear
from Models.autoencoder import cal, Encoder, Decoder, Classifier
from Utility.dataloader import dataloader
from Utility.utility import manual_label_encoder, get_classes
import pandas as pd
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    parser.add_argument('--category', default=None, type=str, required=True) #category (Ex. emotion, race, etc.)
    parser.add_argument("--model_load_and_save_path", default=None, type=str, required=True)
    parser.add_argument("--train_csv", default=f"race_train.csv", type=str, required=False)  # train_csv
    parser.add_argument("--val_csv", default=f"race_val.csv", type=str, required=False)  # val_csv
    parser.add_argument("--test_csv", default=f"race_test.csv", type=str, required=False)  # test_csv
    parser.add_argument("--epochs", default=60, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--learning_rate", default=1e-3, type=int, required=False)

    args = parser.parse_args()
    category = args.category
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv
    PATH_SAVE = args.model_load_and_save_path
    csv_load_path = args.csv_load_path

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

    IMAGE_SIZE = 128
    num_layers = 3
    d = 64

    train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
    val_loader = dataloader(val_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
    test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

    jj, kk = cal(IMAGE_SIZE, num_layers)
    encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

    encoder.load_state_dict(torch.load(PATH_SAVE + "encoder_{}_layers.pt".format(num_layers)))
    cnn = Classifier(encoder, d, OUTPUTS_a)

    model_name = f'cnn_classifier_{category}.pt'

    print("Training...")
    train_and_test(cnn, train_loader, val_loader, classes, model_name, PATH_SAVE, epochs, batch_size, learning_rate)

    print()
    print("Evaluating...")
    print()

    evaluate_best_model(cnn, test_loader, classes, model_name, PATH_SAVE)
