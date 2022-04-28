# About the Data

Accent detection data comes from the following Kaggle archive: https://www.kaggle.com/rtatman/speech-accent-archive?select=recordings. "This dataset contains 2140 speech samples, and speakers are come from 177 countries and have 214 different native languages. Each speaker speaks same content in English." The size of this dataset is 951 MB. In this project, we will focus on top 5 native lanuages and the total number of original data files for those 5 native lanuages is 971. And they are English, Arabic, Spanish, French, and Madarin. The data distribution is showing as below:

* English: 579
* Spanish: 162
* Arabic:  102
* Mandarin: 65
* French: 63

# Acessing and Preprocessing the Data

### Retrieving the Data

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/Data Preprocessing/generate_csv.py" target="_blank">Data Preprocessing</a>
Preprocess the data and create a new csv file to load the audios.  

#### Example: 

```
python3 -m Import_Data --csv_folder "/home/ubuntu/Capstone/Data_Preprocessing/" 
                       --audio_folder "/home/ubuntu/Capstone/recordings/recordings/"
                       --csv_name "speakers_all.csv"
```
 - _csv_folder_: this is the folder of the csv file
 - _audio_folder_: this is the folder of the audio files
 - _csv_name_: this is the file name

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/Data Preprocessing/Trim_Split_Melspec.py" target="_blank">Split and Generating Mel Spectrograms</a>
3. Since the audios are too long, we need to split into chunks put into models. Besides, we generate mel-spectrograms for CNN models.  
### Example:
```
python3 -m Data_Preprocessing.Trim_Split_Melspec.
--csv_folder "/home/ubuntu/Capstone/Data_Preprocessing/"
--save_path "/home/ubuntu/Capstone/Data_Preprocessing/"
```
 - _csv_folder_: this is the folder of the csv file
 - _save_path_: the path to save new audio files, mel-spectrograms and new csv files


# Augmentation

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/Data Processing/Augmentation.py" target="_blank">Augmenting Audio</a>

Augmentation is a method of generating more data by manipulating and distorting the audio to make it slightly different than the original audio. This creates more data in the training set. This file will output 3 folders: augmented_audio which where all the augmented audio files will be stored, augmented_csv where Augmented_Audio_Train.csv and Augmented_Images_Train.csv are stored and augmented_images where all the augmented Mel Spectrograms are stored. The Augmented_Images_Train.csv can be used to replace emotion_train.csv for any training algorithms. 

To run augmentation.py, use the follow command as reference:

```
python3 -m Augmentation.py --csv_folder "/home/ubuntu/Capstone/Data_Preprocessing/"                    
                        --csv_name "new_train_trim_split_spec.csv"               
                        --audio_folder "/home/ubuntu/Capstone/recordings/recordings/"         
                        --save_path "/home/ubuntu/Capstone/data_train_val_test/"
```
- _csv_folder_: path to load in the train data 
- _csv_name_: training data file name
- _audio_folder_: path to load original audio files
- _save_path_: path to save augmented audio files and images

After user finished the data augmentation, user needs to combine the origin data (audio/images) with the augmented data (audio/images).
To run Combine_Origin_Augmentated.py, use the follow command as reference:
``` 
python3 -m Combine_Origin_Augmentated --origin_csv_path "/home/ubuntu/Capstone/data_train_val_test/" 
                         --augmented_csv_path "/home/ubuntu/Capstone/data_csv_0326/augmented_csv/" 
                         --augmented_data_path "/home/ubuntu//Capstone/data_csv_0326/augmented_images"
                         --combined_csv_path "."
                         --origin_csv_name "new_train_trim_split_spec.csv" 
                         --augmented_csv_name "Augmented_Images.csv"

```
- _origin_csv_path_: path to original csv file
- _augmented_csv_path_: path to augmented csv file
- _augmented_data_path_: path to load augmented data
- _combined_csv_path_: path to save combined csv file
- _origin_csv_name_: original csv file name
- _augmented_csv_name_: augmented csv file name
 
 
# Running CNN Models

There are several CNN models that are implemented in this repository. We designed a 3 layer and 9 layer CNN model and have options for running several pretrained models: Resnet18, Resnet34, VGG16, EfficientNet_b2. 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/CNN/Train/Train_CNN.py" target="_blank">Training CNN</a>

To execute these models, run Train_CNN.py with the following arguments: 

#### Example: 

```
python3 -m Models.Train_CNN --csv_load_path "/home/ubuntu/Capstone/data_train_val_test/" 
                            --val_csv "new_val_trim_split_spec.csv"  
                            --train_csv "new_train_trim_split_spec.csv" 
                            --test_csv "new_test_trim_split_spec.csv"
                            --model_dir " home/ubuntu/Capstone/CNN/Models/Saved_Models/new_saved_models/"


```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _model_dir_: directory to save the model
- _train_csv_: training dataset
- _val_csv_: validation dataset
- _test_csv_: testing dataset


# Autoencoder

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/CNN/Train/Train_AutoEncoder.py" target="_blank">Training AutoEncoder</a>

The AutoEncoder is used for pretraining the 3 layer CNN model we created. To use the autoencoder, there are 2 steps required: training the autoencoder and loading the autoencoder model parameters to train the classifier to output predictions using the autoencoder model parameters as a starting point.

To run train_autoencoder.py, use the following command as reference:

#### Example

```
python3 -m Train.Train_AutoEncoder --csv_load_path "/home/ubuntu/Capstone/data_csv_0326/" 
                                   --csv_file "data_csv_0326accent_train_trim_split.csv"  
                                   -model_save_path "/home/ubuntu/Capstone/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _csv_file_: training dataset
- _model_save_path_: the folder path to save the the AutoEncoder parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/CNN/Test/Test_AutoEncoder.py" target="_blank">Testing AutoEncoder</a>

The purpose of testing the AutoEncoder is to make sure model parameters that were saved can be loaded again and produce the same results which, in this case, is measured by the loss. 

To run testing_autoencoder.py, use the following command as reference:

#### Example

```
python3 -m Test.Test_AutoEncoder --csv_load_path "/home/ubuntu/Capstone/data_csv_0326/"   
                                 --csv_file "data_csv_0326accent_train_trim_split.csv"   
                                 --model_save_path "/home/ubuntu/Capstone/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train csv file
- _csv_file_: the train csv file 
- _model_save_path_: the folder path to load the the AutoEncoder parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/CNN/Test/CNN_Classifier.py" target="_blank">AutoEncoder for Classification</a>

Once the autoencoder model parameters are saved and tested, the next step is to use them as a baseline for training a classifier using the same architecture as the Encoder with a classification layer at the end matching the number of classes to make predictions on. 

To run cnn_classifier.py, use the following command as reference:

```
python3 -m Test.CNN_Classifier --csv_load_path "/home/ubuntu/Capstone/data_train_val_test/"                   
                     --val_csv "new_val_trim_split_spec.csv"                
                     --train_csv "new_train_trim_split_spec.csv"         
                     --test_csv "new_test_trim_split_spec.csv" 
                     --model_load_and_save_path "/home/ubuntu/Capstone/CNN/Models/Saved_Models/"
```
- _csv_load_path_: folder path to load the train, validation and test csv files
- _train_csv_: the train csv file 
- _val_csv_: the validation csv file 
- _test_csv_: the test csv file 
- _model_load_and_save_path_: the folder path to load and save the the model parameters as a state dict object in pickle format (model.pt)

# Pseudo Labeling

Pseudo labeling is a form of pretraining that allows for the use of unlabeled data; it is primarily used for unbalanced or small datasets. In this repository, unlabeled CREMA-D data is used along with labeled Accent data. 

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/CNN/Models/Pseudo_Labeling.py" target="_blank">Pseudo Labeling using Semi-Supervised Learning</a>

To run pseudo_labeling_semisupervised.py, use the follow command as reference:

#### Example: 

```
python3 -m Models.Pseudo_Labeling --csv_path "/home/ubuntu/Capstone/data_train_val_test/" 
                                  --train_csv "new_train_trim_split_spec.csv" 
                                  --test_csv "new_test_trim_split_spec.csv" 
                                  --val_csv "new_val_trim_split_spec.csv"
                                  --model_path "/home/ubuntu/Capstone/CNN/Models/Saved_Models/" 
                                  --unlabel_csv_path "/home/ubuntu/Capstone/race_data/CREMA-D/" 
                                  --unlabel_csv_name "Mel_Spectrograms.csv"
                                  --model_name "restnet34.pt"
                                  
```
- _csv_path_: folder path to load the train, validation and test csv files
- _train_csv_: the train csv file 
- _val_csv_: the validation csv file 
- _test_csv_: the test csv file 
- _model_path_: this is the path of the model 
- _unlabel_csv_path_: unlabeled data csv path
- _unlabel_csv_name_: unlabeled data csv name
- _model_name_: this is the best model that user saved before


# Transformer

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/Transformers/Training/Training_Wav2Vec2.py" target="_blank">Training on Wav2Vec 2.0</a>

Wav2Vec 2.0 is a powerful framework for self-supervised learning of speech representations. This model produced better results training on accent data than all CNN models used in this repository.

#### Example: 

```
python3 -m Training.Training_Wav2Vec2 --csv_load_path "/home/ubuntu/Capstone/data_train_val_test/"                 
                             --train_csv "new_train_trim_split_audio.csv"                 
                             --val_csv "new_val_trim_split_audio.csv"
                             --model_path "Capstone/Finalized_Code/Transformer/"

```                                        
- _csv_load_path_: folder path to load the train, validation and test csv files
- _train_csv_: the train csv file 
- _val_csv_: the validation csv file 
- _model_path_: the folder path to save the the model parameters as a state dict object in pickle format (model.pt)

### <a href="https://github.com/sz389/Capstone-Group5/blob/main/Code/Accent/Transformer/Testing/Evaluate_Transformer.py" target="_blank">Evaluating on Wav2Vec 2.0</a>

This code file will evaluate the best model checkpoint produced from training_wav2vec2.py on the test set to get the final results.

#### Example: 

```
python3 -m Testing.Evaluate_Transformer --csv_load_path "/home/ubuntu/Capstone/data_train_val_test/"                 
                             --test_csv "new_test_trim_split_audio.csv"              
                             --model_path "/home/ubuntu/Capstone/saved_model/"   
                             --model_name "wav2vec2-large-960h-finetuned-ks/checkpoint-2232"
``` 

- _csv_load_path_: folder path to load the test csv files
- _test_csv_: the test csv file 
- _model_path_: the folder path to load the the best model parameters from training_wav2vec2.py as a state dict object in pickle format (model.pt)
- _model_name_: the best model saved from Wav2Vec validation 
