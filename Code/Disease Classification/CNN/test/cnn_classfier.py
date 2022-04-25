import torch
from CNN.models.cnn import CNN, train_and_test, evaluate_best_model, add_linear, combine_and_evaluate
from CNN.models.autoencoder import cal, Encoder, Decoder, Classifier
from utility import dataloader,get_n_params
import pandas as pd
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_load_path", default=None, type=str, required=True)
    # parser.add_argument('--category', default=None, type=str, required=True) #category (Ex. emotion, race, etc.)

    # category = args.category

    # parser.add_argument("--train_csv", default=f"{category}_train.csv", type=str, required=False)  # train_csv
    # parser.add_argument("--val_csv", default=f"{category}_val.csv", type=str, required=False)  # val_csv
    # parser.add_argument("--test_csv", default=f"{category}_test.csv", type=str, required=False)  # test_csv

    parser.add_argument("--epochs", default=60, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--learning_rate", default=1e-3, type=int, required=False)

    parser.add_argument("--model_load_and_save_path", default=None, type=str, required=True)

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    # train_csv = args.train_csv
    # val_csv = args.val_csv
    # test_csv = args.test_csv
    PATH_SAVE = args.model_load_and_save_path
    csv_load_path = args.csv_load_path

    train_df = pd.read_csv(csv_load_path + "/KCL_train_trim_split_spec.csv")
    val_df = pd.read_csv(csv_load_path + "/KCL_valid_trim_split_spec.csv")
    test_df = pd.read_csv(csv_load_path + "/KCL_test_trim_split_spec.csv")

    classes = ['hc','pd']
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