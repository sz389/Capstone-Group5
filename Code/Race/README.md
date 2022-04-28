# About the Data

Race classification, for the purpose of this study, will consist of classification between male and female voices. 
The dataset that is being used in this model comes from a Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D). The data contains  age, sex, race, ethnicity information from 91 actors, 48 male and 42 female actors who speak 12 sentences in 6 emotions at 4 emotion levels. The data comes from this repository: https://github.com/CheyneyComputerScience/CREMA-D

The dataset contains metadata about each audio file: 

![image](https://user-images.githubusercontent.com/54903276/164991574-2c61d7ba-382b-4095-8443-2bc6cd93e742.png)

The class distribution is as follows: 

![image](https://user-images.githubusercontent.com/54903276/152845505-2a46bc3e-765c-4f71-aedd-f2df60bc3481.png)

Along with ethnicity (which was not used in this study):

![image](https://user-images.githubusercontent.com/54903276/152845609-725bd426-3ef4-4532-8b43-2c241519bf4b.png)

# Accessing and Preprocessing the Data

### Retrieving the Data

1. Follow the instructions on CREMA-D Github README to access the AudioWAV folder to retrieve all the audio files.

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/Data%20Processing/data_preprocessing.py" target="_blank">Data Preprocessing</a>
2. Using the CREMA_no_path.csv file that can be found in the Data folder, along with the AudioWAV folder, run data_preprocessing.py in the terminal in the following format to generate a new csv file with the audio file name, audio path and corresponding labels.

#### Example: 

```
python3 data_preprocessing.py --csv_file "CREMA_no_path.csv"                     
                              --csv_path "/Race/Data/"                 
                              --audio_path "/Race/Data/AudioWAV/"              
                              --category "race"                    
```
 - _csv_file_: this is the csv file found in the Data folder
 - _csv_path_: the path to the csv file
 - _audio_path_: the path to the AudioWAV folder downloaded from the CREMA-D github
 - _category_: either sex, age, race or emotion
 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/Data%20Processing/generate_mel_spectrogram.py" target="_blank">Generating Mel Spectrograms</a>
3. Using the csv file generated from data_preprocessing.py called CREMA_with_audio_path.csv, run generate_mel_spectrograms.py to generate Mel Spectrograms in a user-defined folder along with train, validation and test csv files which all include a column with the image path.

# Running CNN Models

There are several CNN models that are implemented in this repository. We designed a 3 layer and 9 layer CNN model and have options for running several pretrained models: Resnet18, Resnet34, VGG16, EfficientNet_b2. 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/CNN/Training/train_cnn.py" target="_blank">Training CNN</a>

To execute these models, run train_cnn.py with the following arguments: 

#### Example: 

```
python3 train_cnn.py --csv_load_path "/Race/Data/"                    
                     --category "race"               
                     --train_csv "race_train.csv"          
                     --val_csv "race_val.csv"
                     --test_csv "race_test.csv"
                     --epochs 30
                     --batch_size 64
                     --learning_rate 1e-3
                     --model "resnet18"
                     --model_save_path "/Race/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = race_train.csv)
- _val_csv_: the validation csv file (default = race_val.csv)
- _test_csv_: the test csv file (default = race_test.csv)
- _epochs_: the number of epochs the model should run for (default = 30)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_: one of "cnn3", "cnn9", "resnet18", "renset34", "vgg16", "efficientnet"
- _model_save_path_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)


# Augmentation

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/Data_Processing/augmentation.py" target="_blank">Augmenting Audio</a>

Augmentation is a method of generating more data by manipulating and distorting the audio to make it slightly different than the original audio. This creates more data in the training set. This file will output 3 folders: augmented_audio which where all the augmented audio files will be stored, augmented_csv where Augmented_Audio_Train.csv and Augmented_Images_Train.csv are stored and augmented_images where all the augmented Mel Spectrograms are stored. The Augmented_Images_Train.csv can be used to replace emotion_train.csv for any training algorithms. 

To run augmentation.py, use the follow command as reference:

```
python3 augmentation.py --csv_folder "/Race/Data/"                    
                        --save_path "/Race/Data/"               
                        --category "race"      
```
- _csv_folder_: path to to load in the train data (race_train.csv)
- _save_path_: where the augmented Audio and Mel Spectrograms should be saved
- _category_: either "sex", "age", "race", "emotion"


# Autoencoder

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/CNN/Training/train_autoencoder.py" target="_blank">Training AutoEncoder</a>

The AutoEncoder is used for pretraining the 3 layer CNN model we created. To use the autoencoder, there are 3 steps required: training the autoencoder, testing the autoencoder and loading the autoencoder model parameters to train the classifier to output predictions using the autoencoder model parameters as a starting point.

To run train_autoencoder.py, use the following command as reference:

#### Example

```
python3 train_autoencoder.py --csv_load_path "/Race/Data/"                    
                             --category "race"               
                             --train_csv "race_train.csv"          
                             --epochs 200
                             --batch_size 64
                             --learning_rate 1e-3
                             --model_save_path "/Race/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = race_train.csv)
- _epochs_: the number of epochs the model should run for (default = 200)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_path_: the folder path to save the the AutoEncoder parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/CNN/Testing/testing_autoencoder.py" target="_blank">Testing AutoEncoder</a>

The purpose of testing the AutoEncoder is to make sure model parameters that were saved can be loaded again and produce the same results which, in this case, is measured by the loss. 

To run testing_autoencoder.py, use the following command as reference:

#### Example

```
python3 testing_autoencoder.py --csv_load_path "/Race/Data/"                    
                               --category "race"               
                               --train_csv "race_train.csv"          
                               --epochs 5
                               --batch_size 64
                               --learning_rate 1e-3
                               --model_load_path "/Race/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = race_train.csv)
- _epochs_: the number of epochs the model should run for (default = 5)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_path_: the folder path to load the the AutoEncoder parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/CNN/Testing/cnn_classifier.py" target="_blank">AutoEncoder for Classification</a>

Once the autoencoder model parameters are saved and tested, the next step is to use them as a baseline for training a classifier using the same architecture as the Encoder with a classification layer at the end matching the number of classes to make predictions on. 

To run cnn_classifier.py, use the following command as reference:

```
python3 train_cnn.py --csv_load_path "/Race/Data/"                    
                     --category "race"               
                     --train_csv "race_train.csv"          
                     --val_csv "race_val.csv"
                     --test_csv "race_test.csv"
                     --epochs 60
                     --batch_size 64
                     --learning_rate 1e-3
                     --model_save_and_load_path "/Race/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = race_train.csv)
- _val_csv_: the validation csv file (default = race_val.csv)
- _test_csv_: the test csv file (default = race_test.csv)
- _epochs_: the number of epochs the model should run for (default = 60)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_and_load_path_: the folder path to load and save the the model parameters as a state dict object in pickle format (model.pt)

# Pseudo Labeling

Pseudo labeling is a form of pretraining that allows for the use of unlabeled data; it is primarily used for unbalanced or small datasets. In this repository, unlabeled Accent data is used along with labeled Emotions data. 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/Data%20Processing/Pseudo%20Labeling/pseudo_labeling_semisupervised.py" target="_blank">Pseudo Labeling using Semi-Supervised Learning</a>

To run pseudo_labeling_semisupervised.py, use the follow command as reference:

#### Example: 

```
python3 pseudo_labeling_semisupervised.py --csv_load_path "/Race/Data/"                    
                                          --category "race"
                                          --model "resnet18"
                                          --cnn_param_file "resnet18_race.pt"
                                          --pseudolabeling_param_file "resnet18_race_PL.pt"
                                          --model_save_and_load_path "/Race/CNN/Models/Saved_Models/"
                                          --train_csv "race_train.csv"          
                                          --val_csv "race_val.csv"
                                          --test_csv "race_test.csv"
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
- _train_csv_: the train csv file (default = race_train.csv)
- _val_csv_: the validation csv file (default = race_val.csv)
- _test_csv_: the test csv file (default = race_test.csv)
- _unlabeled_csv_: the csv file for the unlabeled data
- _epochs_: the number of epochs the model should run for (default = 150)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)


# Transformer

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/Transformer/Training/training_wav2vec2.py" target="_blank">Training on Wav2Vec 2.0</a>

Wav2Vec 2.0 is a powerful framework for self-supervised learning of speech representations. This model produced better results training on emotion data than all CNN models used in this repository.

#### Example: 

```
python3 training_wav2vec2.py --csv_load_path "/Race/Data/"                    
                             --category "race"                             
                             --model_save_name "transformer_race"
                             --model_save_path "/Race/Transformer/Models/Saved_Models/"
                             --train_csv "race_train.csv"          
                             --val_csv "race_val.csv"
                             --epochs 20
                             --batch_size 4
                             --learning_rate 3e-5

```                                        
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = race_train.csv)
- _val_csv_: the validation csv file (default = race_val.csv)
- _epochs_: the number of epochs the model should run for (default = 20)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 3e-5)
- _model_save_name_: the name of the folder the model checkpoint should be saved in
- _model_save_path_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Race/Transformer/Testing/evaluate_wav2vec2.py" target="_blank">Evaluating on Wav2Vec 2.0</a>

This code file will evaluate the best model checkpoint produced from training_wav2vec2.py on the test set to get the final results.

#### Example: 

```
python3 evaluate_wav2vec2.py --csv_load_path "/Race/Data/"                    
                             --category "race"               
                             --test_csv "race_test.csv"          
                             --epochs 20
                             --batch_size 4
                             --learning_rate 3e-5
                             --model_load_path "/Race/Transformer/Saved_Models/wav2vec2-base-finetuned-ks/checkpoint-835”
                             --model_dir_path "/Race/Transformer/Saved_Models/"
``` 

- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _test_csv_: the test csv file (default = race_test.csv)
- _epochs_: the number of epochs the model should run for (default = 20)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 3e-5)
- _model_save_path_: the folder path to load the the best model parameters from training_wav2vec2.py as a state dict object in pickle format (model.pt)
- _model_dir_path_: a directory created by the Trainer class 



