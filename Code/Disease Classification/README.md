# About the Data

Disease Classification for this project will consist of interpreting audio files in which healthy and Parkinson's patients spontaneouly talk or read text. The notebook in the code folder is written in Python and uses CNN model and Wav2Vec2 transformer with a classification head to train a dataset.   

The dataset reference:  
_Hagen Jaeger, Dhaval Trivedi, & Michael Stadtschnitzer. (2019). Mobile Device Voice Recordings at King's College London (MDVR-KCL) from both early and advanced Parkinson's disease patients and healthy controls [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2867216_  

_Keith Ito &  Linda Johnson. (2017). The LJ Speech Dataset. LJspeech17. https://keithito.com/LJ-Speech-Dataset/_


# Accessing and Preprocessing the Data

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


### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Disease%20Classification/CNN/test/cnn_classfier.py" target="_blank">AutoEncoder for Classification</a>

Once the autoencoder model parameters are saved and tested, the next step is to use them as a baseline for training a classifier using the same architecture as the Encoder with a classification layer at the end matching the number of classes to make predictions on. 

To run cnn_classifier.py, use the following command as reference:

```
python3 -m CNN.test.cnn_classfier --csv_load_path "/home/ubuntu/Capstone/data/"             
                                  --epochs 60
                                  --batch_size 64
                                  --learning_rate 1e-3
                                  --model_save_and_load_path "/home/ubuntu/Capstone/saved_model/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _epochs_: the number of epochs the model should run for (default = 60)
- _batch_size_: the batch size for the dataloader (default = 64)
- _learning_rate_: the learning rate of the model (default = 1e-3)
- _model_save_and_load_path_: the folder path to load and save the the model parameters as a state dict object in pickle format (model.pt)

# Pseudo Labeling

Pseudo labeling is a form of pretraining that allows for the use of unlabeled data; it is primarily used for unbalanced or small datasets. In this repository, unlabeled Accent data is used along with labeled Emotions data. 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Disease%20Classification/CNN/train/pseudolabel.py" target="_blank">Pseudo Labeling using Semi-Supervised Learning</a>

To run pseudo_labeling_semisupervised.py, use the follow command as reference:

#### Example: 

```
python3 -m CNN.train.pseudolabel --csv_load_path "/home/ubuntu/Capstone/data/" 
                                 --epochs 50 
                                 --model_dir "/home/ubuntu/Capstone/saved_model/" 
                                 --unlabeled_csv "/home/ubuntu/Capstone/LJSpeech-1.1/LJimg.csv"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _unlabeled_csv_: the csv file for the unlabeled data
- _model_dir_: the model directory to load the model
- _epochs_: the number of epochs the model should run for (default = 150)


# Transformer

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Disease%20Classification/Transformer/Wav2vec2.py" target="_blank">Training on Wav2Vec 2.0</a>

Wav2Vec 2.0 is a powerful framework for self-supervised learning of speech representations. This model produced better results training on emotion data than all CNN models used in this repository.

#### Example: 

```
python3 -m Transformer.Wav2vec2 --csv_folder "/home/ubuntu/Capstone/data/" 
                                --model_dir "/home/ubuntu/Capstone/saved_model/"
```                                        
- _csv_folder_: folder path to load the train, validation and test csv files
- _model_dir_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Disease%20Classification/Transformer/evaluate_transformer.py" target="_blank">Evaluating on Wav2Vec 2.0</a>

This code file will evaluate the best model checkpoint produced from training_wav2vec2.py on the test set to get the final results.

#### Example: 

```
python3 -m Transformer.evaluate_transformer --csv_folder "/home/ubuntu/Capstone/data/" 
                                            --model_dir "/home/ubuntu/Capstone/saved_model/"
                                            --checkpoint_num 150
``` 

- _csv_folder_: folder path to load the train, validation and test csv files
- _model_dir_: the folder path to load the the best model 
- _checkpoint_num_: the best model check point (default=150) (eg: if the checkpoint number is 150, the model will load from model_dir + "/wav2vec2-base-finetuned-ks/checkpoint-150/"
