# About the Data

Emotion Classification for this project will consist of interpreting audio files in which sentences are read in an angry, happy, sad, neutral, disgusted or fear. The notebook in the code folder is written in Python and uses a transformer called Wav2Vec2 with a classification head to train a dataset. The dataset that is being used in this model comes from a Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D). The data contains  age, sex, race, ethnicity information from 91 actors, 48 male and 42 female actors who speak 12 sentences in 6 emotions at 4 emotion levels. The data comes from this repository: https://github.com/CheyneyComputerScience/CREMA-D

The dataset used contains metadata about each audio file:
![image](https://user-images.githubusercontent.com/54903276/152839639-2366c610-afdc-41cb-92a3-c21fef91c929.png)

The value counts of the emotions to look at the class distribution:

![image](https://user-images.githubusercontent.com/54903276/152840702-e469632d-4b65-4992-8d71-4a2fcbff199a.png)


# Acessing and Preprocessing the Data

### Retrieving the Data

1. Follow the instructions on CREMA-D Github README to access the AudioWAV folder to retrieve all the audio files.

### Data Preprocessing
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
 

### Generating Mel Spectrograms
3. Using the csv file generated from data_preprocessing.py called CREMA_with_audio_path.csv, run generate_mel_spectrograms.py to generate Mel Spectrograms in a user-defined folder along with train, validation and test csv files which all include a column with the image path.

# Running CNN Models

There are several CNN models that are implemented in this repository. We designed a 3 layer and 9 layer CNN model and have options for running several pretrained models: Resnet18, Resnet34, VGG16, EfficientNet_b2. 

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
- _train_csv_: the train csv file (not required)
- _val_csv_: the validation csv file (not required)
- _test_csv_: the test csv file (not required)
- _epochs_: the number of epochs the model should run for (not required)
- _batch_size_: the batch size for the dataloader (not required)
- _learning_rate_: the learning rate of the model (not required)
- _model_: one of "cnn3", "cnn9", "resnet18", "renset34", "vgg16", "efficientnet"
- _model_save_path_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)


# Using the AutoEncoder

The AutoEncoder is used for pretraining the 3 layer CNN model we created. To use the autoencoder, there are 3 steps required: training the autoencoder, testing the autoencoder and loading the autoencoder model parameters to train the classifier to output predictions using the autoencoder model parameters as a starting point.

To run train_autoencoder.py, use the following arguments:
