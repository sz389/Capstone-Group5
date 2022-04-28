# About the Data

Disease Classification for this project will consist of interpreting audio files in which healthy and Parkinson's patients spontaneouly talk or read text. The notebook in the code folder is written in Python and uses CNN model and Wav2Vec2 transformer with a classification head to train a dataset.   
The dataset reference:  
Hagen Jaeger, Dhaval Trivedi, & Michael Stadtschnitzer. (2019). Mobile Device Voice Recordings at King's College London (MDVR-KCL) from both early and advanced Parkinson's disease patients and healthy controls [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2867216


# Acessing and Preprocessing the Data

### Retrieving the Data

1. Download the data.  
First change directory to the place you want to download the data. Then run the following code to download the data.

```
wget https://zenodo.org/record/2867216/files/26_29_09_2017_KCL.zip
unzip 26_29_09_2017_KCL.zip
```

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Disease%20Classification/Data_Preprocessing/generate_csv.py" target="_blank">Data Preprocessing</a>
2. Preprocess the data and create a new csv file to load the audios.  
Previous to run the code, change the directory to root directory of this category (eg: cd path/Disease Classification)  

#### Example: 

```
python3 -m Data_Preprocessing.generate_csv --load_path "/home/ubuntu/Capstone/data/original_files/26-29_09_2017_KCL/" 
                                           --save_path "/home/ubuntu/Capstone/data/"                             
```
 - _load_path_: this is the data folder
 - _save_path_: the folder to save the csv file
 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Disease%20Classification/Data_Preprocessing/trim_split.py" target="_blank">Generating Mel Spectrograms</a>
3. Since the audios are too long, we need to split into chunks put into models. Besides, we generate mel-spectrograms for CNN models.  
### Example:
```
python3 -m Data_Preprocessing.trim_split --csv_folder "/home/ubuntu/Capstone/data/"
                                         --save_path "/home/ubuntu/Capstone/data/"
```
 - _csv_folder_: this the folder of the csv file
 - _save_path_: the path to save mel-spectrograms and new csv files
 


# Running CNN Models

There are several CNN models that are implemented in this repository. We designed a 3 layer and 9 layer CNN model and have options for running several pretrained models: Resnet18, Resnet34, VGG16, EfficientNet_b2. 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Disease%20Classification/CNN/train/train_cnn.py" target="_blank">Training CNN</a>

To execute these models, run train_cnn.py with the following arguments: 

#### Example: 

```
python3 -m CNN.train.train_cnn --csv_folder "/home/ubuntu/Capstone/data/" 
                               --model_dir "/home/ubuntu/Capstone/saved_model/"

```
- _csv_folder_: folder path to load the train, validation and test csv files
- _model_dir_: directory to save the model




# Autoencoder

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Disease%20Classification/CNN/train/train_autoencoder.py" target="_blank">Training AutoEncoder</a>

The AutoEncoder is used for pretraining the 3 layer CNN model we created. To use the autoencoder, there are 2 steps required: training the autoencoder and loading the autoencoder model parameters to train the classifier to output predictions using the autoencoder model parameters as a starting point.

To run train_autoencoder.py, use the following command as reference:

#### Example

```
python3 -m CNN.train.train_autoencoder --csv_load_path "/home/ubuntu/Capstone/data/"          
                                       --epochs 200
                                       --batch_size 64
                                       --learning_rate 1e-3
                                       --model_save_path "/home/ubuntu/Capstone/saved_model/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _epochs_: the number of epochs the model should run for (default = 200)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_path_: the folder path to save the the AutoEncoder parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/CNN/Testing/testing_autoencoder.py" target="_blank">Testing AutoEncoder</a>

The purpose of testing the AutoEncoder is to make sure model parameters that were saved can be loaded again and produce the same results which, in this case, is measured by the loss. 

To run testing_autoencoder.py, use the following command as reference:

#### Example

```
python3 testing_autoencoder.py --csv_load_path "/Emotions/Data/"                    
                               --category "emotion"               
                               --train_csv "emotion_train.csv"          
                               --epochs 5
                               --batch_size 64
                               --learning_rate 1e-3
                               --model_load_path "/Emotions/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = emotion_train.csv)
- _epochs_: the number of epochs the model should run for (default = 5)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_path_: the folder path to load the the AutoEncoder parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/CNN/Testing/cnn_classifier.py" target="_blank">AutoEncoder for Classification</a>

Once the autoencoder model parameters are saved and tested, the next step is to use them as a baseline for training a classifier using the same architecture as the Encoder with a classification layer at the end matching the number of classes to make predictions on. 

To run cnn_classifier.py, use the following command as reference:

```
python3 train_cnn.py --csv_load_path "/Emotions/Data/"                    
                     --category "emotion"               
                     --train_csv "emotion_train.csv"          
                     --val_csv "emotion_val.csv"
                     --test_csv "emotion_test.csv"
                     --epochs 60
                     --batch_size 64
                     --learning_rate 1e-3
                     --model_save_and_load_path "/Emotions/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = emotion_train.csv)
- _val_csv_: the validation csv file (default = emotion_val.csv)
- _test_csv_: the test csv file (default = emotion_test.csv)
- _epochs_: the number of epochs the model should run for (default = 60)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_and_load_path_: the folder path to load and save the the model parameters as a state dict object in pickle format (model.pt)

# Pseudo Labeling

Pseudo labeling is a form of pretraining that allows for the use of unlabeled data; it is primarily used for unbalanced or small datasets. In this repository, unlabeled Accent data is used along with labeled Emotions data. 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/Data%20Processing/Pseudo%20Labeling/pseudo_labeling_semisupervised.py" target="_blank">Peudo Labeling using Semi-Supervised Learning</a>

To run pseudo_labeling_semisupervised.py, use the follow command as reference:

#### Example: 

```
python3 pseudo_labeling_semisupervised.py --csv_load_path "/Emotions/Data/"                    
                                          --category "emotion"
                                          --model "resnet18"
                                          --cnn_param_file "resnet18_emotions.pt"
                                          --pseudolabeling_param_file "resnet18_emotion_PL.pt"
                                          --model_save_and_load_path "/Emotions/CNN/Models/Saved_Models/"
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
- _train_csv_: the train csv file (default = emotion_train.csv)
- _val_csv_: the validation csv file (default = emotion_val.csv)
- _test_csv_: the test csv file (default = emotion_test.csv)
- _unlabeled_csv_: the csv file for the unlabeled data
- _epochs_: the number of epochs the model should run for (default = 150)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)


# Transformer

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/Transformer/Training/training_wav2vec2.py" target="_blank">Training on Wav2Vec 2.0</a>

Wav2Vec 2.0 is a powerful framework for self-supervised learning of speech representations. This model produced better results training on emotion data than all CNN models used in this repository.

#### Example: 

```
python3 training_wav2vec2.py --csv_load_path "/Emotions/Data/"                    
                             --category "emotion"                             
                             --model_save_name "transformer_emotion"
                             --model_save_path "/Emotions/Transformer/Models/Saved_Models/"
                             --train_csv "emotion_train.csv"          
                             --val_csv "emotion_val.csv"
                             --epochs 20
                             --batch_size 4
                             --learning_rate 3e-5

```                                        
- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _train_csv_: the train csv file (default = emotion_train.csv)
- _val_csv_: the validation csv file (default = emotion_val.csv)
- _epochs_: the number of epochs the model should run for (default = 20)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 3e-5)
- _model_save_name_: the name of the folder the model checkpoint should be saved in
- _model_save_path_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Emotions/Transformer/Testing/evaluate_wav2vec2.py" target="_blank">Evaluating on Wav2Vec 2.0</a>

This code file will evaluate the best model checkpoint produced from training_wav2vec2.py on the test set to get the final results.

#### Example: 

```
python3 evaluate_wav2vec2.py --csv_load_path "/Emotions/Data/"                    
                             --category "emotion"               
                             --test_csv "emotion_test.csv"          
                             --epochs 20
                             --batch_size 4
                             --learning_rate 3e-5
                             --model_load_path "/Emotions/Transformer/Saved_Models/wav2vec2-base-finetuned-ks/checkpoint-835”
                             --model_dir_path "/Emotions/Transformer/Saved_Models/"
``` 

- _csv_load_path_: folder path to load the train, validation and test csv files
- _category_: either "sex", "age", "race", "emotion"
- _test_csv_: the test csv file (default = emotion_test.csv)
- _epochs_: the number of epochs the model should run for (default = 20)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 3e-5)
- _model_save_path_: the folder path to load the the best model parameters from training_wav2vec2.py as a state dict object in pickle format (model.pt)
- _model_dir_path_: a directory created by the Trainer class 
