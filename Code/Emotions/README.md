# About the Data

Emotion Classification for this project will consist of interpreting audio files in which sentences are read in an angry, happy, sad, neutral, disgusted or fear. The notebook in the code folder is written in Python and uses a transformer called Wav2Vec2 with a classification head to train a dataset. The dataset that is being used in this model comes from a Crowd-sourced Emotional Multimodal Actors Dataset <a href="https://github.com/CheyneyComputerScience/CREMA-D" target="_blank">(CREMA-D)</a>. The data contains  age, sex, race, ethnicity information from 91 actors, 48 male and 42 female actors who speak 12 sentences in 6 emotions at 4 emotion levels. 

The dataset contains metadata about each audio file: 

![image](https://user-images.githubusercontent.com/54903276/164991574-2c61d7ba-382b-4095-8443-2bc6cd93e742.png)

The class distribution of emotions is shown in the table:

![image](https://user-images.githubusercontent.com/54903276/152840702-e469632d-4b65-4992-8d71-4a2fcbff199a.png)

# Acessing and Preprocessing the Data

### Retrieving the Data

1. Follow the instructions on CREMA-D Github README to access the AudioWAV folder to retrieve all the audio files.

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/Data%20Processing/data_preprocessing.py" target="_blank">Data Preprocessing</a>
2. Using the CREMA_no_path.csv file that can be found in the Data folder, along with the AudioWAV folder, run data_preprocessing.py in the terminal in the following format to generate a new csv file with the audio file name, audio path and corresponding labels.

#### Example: 

```
python3 data_preprocessing.py --csv_file "CREMA_no_path.csv"                     
                              --csv_path "/capstone/Data/"                 
                              --audio_path "/capstone/Data/AudioWAV/"              
                              --category "sex"                    
```
 - _csv_file_: this is the csv file found in the Data folder
 - _csv_path_: the path to the csv file
 - _audio_path_: the path to the AudioWAV folder downloaded from the CREMA-D github
 - _category_: either sex, age, race or emotion
 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/Data%20Processing/generate_mel_spectrogram.py" target="_blank">Generating Mel Spectrograms</a>
3. Using the csv file generated from data_preprocessing.py called CREMA_with_audio_path.csv, run generate_mel_spectrograms.py to generate Mel Spectrograms in a user-defined folder along with train, validation and test csv files which all include a column with the image path.

# Running CNN Models

There are several CNN models that are implemented in this repository. We designed a 3 layer and 9 layer CNN model and have options for running several pretrained models: Resnet18, Resnet34, VGG16, EfficientNet_b2. 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/CNN/Training/train_cnn.py" target="_blank">Training CNN</a>

To execute these models, run train_cnn.py with the following arguments: 

#### Example: 

```
python3 train_cnn.py --csv_load_path "Emotion/Data/"                    
                     --category "emotion"               
                     --train_csv "emotion_train.csv"          
                     --val_csv "emotion_val.csv"
                     --test_csv "emotion_test.csv"
                     --epochs 30
                     --batch_size 64
                     --learning_rate 1e-3
                     --model "resnet18"
                     --model_save_path "Emotion/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = {category}_train.csv)
- _val_csv_: the validation csv file (default = {category}_val.csv)
- _test_csv_: the test csv file (default = {category}_test.csv)
- _epochs_: the number of epochs the model should run for (default = 30)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_: one of "cnn3", "cnn9", "resnet18", "renset34", "vgg16", "efficientnet"
- _model_save_path_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)


# Using the AutoEncoder

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/CNN/Training/train_autoencoder.py" target="_blank">Training AutoEncoder</a>

The AutoEncoder is used for pretraining the 3 layer CNN model we created. To use the autoencoder, there are 3 steps required: training the autoencoder, testing the autoencoder and loading the autoencoder model parameters to train the classifier to output predictions using the autoencoder model parameters as a starting point.

To run train_autoencoder.py, use the following command as reference:

#### Example

```
python3 train_autoencoder.py --csv_load_path "Emotion/Data/"                    
                             --category "emotion"               
                             --train_csv "emotion_train.csv"          
                             --epochs 200
                             --batch_size 64
                             --learning_rate 1e-3
                             --model_save_path "Emotion/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = {category}_train.csv)
- _epochs_: the number of epochs the model should run for (default = 200)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_path_: the folder path to save the the AutoEncoder parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/CNN/Testing/testing_autoencoder.py" target="_blank">Testing AutoEncoder</a>

The purpose of testing the AutoEncoder is to make sure model parameters that were saved can be loaded again and produce the same results which, in this case, is measured by the loss. 

To run testing_autoencoder.py, use the following command as reference:

#### Example

```
python3 testing_autoencoder.py --csv_load_path "Emotion/Data/"                    
                               --category "emotion"               
                               --train_csv "emotion_train.csv"          
                               --epochs 5
                               --batch_size 64
                               --learning_rate 1e-3
                               --model_load_path "Emotion/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = {category}_train.csv)
- _epochs_: the number of epochs the model should run for (default = 5)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_path_: the folder path to load the the AutoEncoder parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/CNN/Testing/cnn_classifier.py" target="_blank">AutoEncoder for Classification</a>

Once the autoencoder model parameters are saved and tested, the next step is to use them as a baseline for training a classifier using the same architecture as the Encoder with a classification layer at the end matching the number of classes to make predictions on. 

To run cnn_classifier.py, use the following command as reference:

```
python3 train_cnn.py --csv_load_path "Emotion/Data/"                    
                     --category "emotion"               
                     --train_csv "emotion_train.csv"          
                     --val_csv "emotion_val.csv"
                     --test_csv "emotion_test.csv"
                     --epochs 60
                     --batch_size 64
                     --learning_rate 1e-3
                     --model_save_and_load_path "Emotion/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = {category}_train.csv)
- _val_csv_: the validation csv file (default = {category}_val.csv)
- _test_csv_: the test csv file (default = {category}_test.csv)
- _epochs_: the number of epochs the model should run for (default = 60)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_and_load_path_: the folder path to load and save the the model parameters as a state dict object in pickle format (model.pt)

# <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/Data%20Processing/Pseudo%20Labeling/pseudo_labeling_semisupervised.py" target="_blank">Pseudo Labeling</a>
Pseudo labeling is a form of pretraining that allows for the use of unlabeled data; it is primarily used for unbalanced or small datasets. In this repository, unlabeled Accent data is used along with labeled Emotions data. 

To run pseudo_labeling_semisupervised.py, use the follow command as reference:

#### Example: 

```
python3 pseudo_labeling_semisupervised.py --csv_load_path "Emotion/Data/"                    
                                          --category "emotion"
                                          --model "resnet18"
                                          --cnn_param_file "resnet18_emotions.pt"
                                          --pseudolabeling_param_file "resnet18_emotion_PL.pt"
                                          --model_save_and_load_path "Emotion/CNN/Models/Saved_Models/"
                                          --train_csv "emotion_train.csv"          
                                          --val_csv "emotion_val.csv"
                                          --test_csv "emotion_test.csv"
                                          --unlabeled_csv "unlabeled.csv"
                                          --epochs 150
                                          --batch_size 64
                                          --learning_rate 1e-3

```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _model_: one of "cnn3", "cnn9", "resnet18", "renset34", "vgg16", "efficientnet"
- _cnn_param_file_: the cnn model.pt file to load 
- _pseudolabeling_param_file_: the model.pt file name to save after semisupervised training to use for evaluation
- _model_save_and_load_path_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)
- _train_csv_: the train csv file (default = {category}_train.csv)
- _val_csv_: the validation csv file (default = {category}_val.csv)
- _test_csv_: the test csv file (default = {category}_test.csv)
- _unlabeled_csv_: the csv file for the unlabeled data
- _epochs_: the number of epochs the model should run for (default = 150)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)


# Transformer

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/Transformer/Training/training_wav2vec2.py" target="_blank">Training on Wav2Vec 2.0</a>

Wav2Vec 2.0 is a powerful framework for self-supervised learning of speech representations. This model produced better results training on emotion data than all CNN models used in this repository.

#### Example: 

```
python3 training_wav2vec2.py --csv_load_path "Emotion/Data/"                    
                             --category "emotion"                             
                             --model_save_name "transformer_emotion"
                             --model_save_path "Emotion/Transformer/Models/Saved_Models/"
                             --train_csv "emotion_train.csv"          
                             --val_csv "emotion_val.csv"
                             --epochs 20
                             --batch_size 4
                             --learning_rate 3e-5

```                                        
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = {category}_train.csv)
- _val_csv_: the validation csv file (default = {category}_val.csv)
- _epochs_: the number of epochs the model should run for (default = 20)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 3e-5)
- _model_save_name_: the name of the folder the model checkpoint should be saved in
- _model_save_path_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/Transformer/Testing/evaluate_wav2vec2.py" target="_blank">Evaluating on Wav2Vec 2.0</a>

This code file will evaluate the best model checkpoint produced from training_wav2vec2.py on the test set to get the final results.

#### Example: 

```
python3 evaluate_wav2vec2.py --csv_load_path "/Emotion/Data/"                    
                             --category "emotion"               
                             --test_csv "emotion_test.csv"          
                             --epochs 20
                             --batch_size 4
                             --learning_rate 3e-5
                             --model_load_path "/Emotion/Transformer/Saved_Models/wav2vec2-base-finetuned-ks/checkpoint-835”
                             --model_dir_path "/Emotion/Transformer/Saved_Models/"
``` 

- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _test_csv_: the test csv file (default = {category}_test.csv)
- _epochs_: the number of epochs the model should run for (default = 20)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 3e-5)
- _model_save_path_: the folder path to load the the best model parameters from training_wav2vec2.py as a state dict object in pickle format (model.pt)
- _model_dir_path_: 
