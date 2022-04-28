import torch
from Models.cnn import CNN, train_and_test, evaluate_best_model, combine_and_evaluate
from Models.AutoEncoder import cal, Encoder, Decoder, Classifier
from utility import dataloader,get_n_params
import pandas as pd
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    parser.add_argument("--train_csv", default=None, type=str, required=True)
    parser.add_argument("--test_csv", default=None, type=str, required=True)
    parser.add_argument("--val_csv", default=None, type=str, required=True)

    parser.add_argument("--epochs", default=60, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--learning_rate", default=1e-3, type=int, required=False)

    parser.add_argument("--model_load_and_save_path", default=None, type=str, required=True)

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv
    PATH_SAVE = args.model_load_and_save_path
    csv_load_path = args.csv_load_path

    train_df = pd.read_csv(csv_load_path + train_csv)
    val_df = pd.read_csv(csv_load_path + val_csv)
    test_df = pd.read_csv(csv_load_path + test_csv)

    train_df['label'] = train_df['native_language']
    val_df['label'] = val_df['native_language']
    test_df['label'] = test_df['native_language']

    train_df['id'] = train_df['path']
    val_df['id'] = val_df['path']
    test_df['id'] = test_df['path']

    classes = ['arabic', 'english', 'french', 'mandarin', 'spanish']
    OUTPUTS_a = len(classes)

    IMAGE_SIZE = 128
    num_layers = 3
    d = 64

    train_loader = dataloader(train_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
    val_loader = dataloader(val_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)
    test_loader = dataloader(test_df, OUTPUTS_a = OUTPUTS_a, BATCH_SIZE = batch_size, IMAGE_SIZE=IMAGE_SIZE)

    jj, kk = cal(IMAGE_SIZE, num_layers)
    encoder = Encoder(encoded_space_dim=d, jj=jj, kk=kk).to(device)

    # encoder.load_state_dict(torch.load(PATH_SAVE + "encoder_{}_layers.pt".format(num_layers)))
    cnn = Classifier(encoder, d, OUTPUTS_a)

    model_name = PATH_SAVE+f'/cnn_classifier_{d}.pt'
    train_and_test(cnn, train_loader, val_loader, classes, model_name, epochs, batch_size, learning_rate)
    evaluate_best_model(cnn, test_loader=test_loader, classes=classes, model_path=model_name)
    combine_and_evaluate(cnn=cnn, model_path=model_name, test_loader=test_loader,
                         xdf_dset_test=test_df, class_names=classes)
    print(f'model parameters: {get_n_params(cnn)}')
