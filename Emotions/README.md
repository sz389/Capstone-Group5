# How to Run

### Retrieving the Data

Emotion Classification for this project will consist of interpreting audio files in which sentences are read in an angry, happy, sad, neutral, disgusted or fear. The notebook in the code folder is written in Python and uses a transformer called Wav2Vec2 with a classification head to train a dataset. The dataset that is being used in this model comes from a Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D). The data contains  age, sex, race, ethnicity information from 91 actors, 48 male and 42 female actors who speak 12 sentences in 6 emotions at 4 emotion levels. The data comes from this repository: https://github.com/CheyneyComputerScience/CREMA-D

1. Follow the instructions on CREMA-D Github README to access the AudioWAV folder to retrieve all the audio files
2. Using the CREMA-D_nopath.csv file along with the AudioWAV folder, run data_preprocessing.py in the terminal in the following format to generate a new csv file. 

#### Example: 

python3 data_preprocessing.py --csv_file "CREMA_no_path.csv" --csv_path "/home/ubuntu/capstone/Data/" --audio_path "/home/ubuntu/capstone/AudioWAV/" --image_path "/home/ubuntu/capstone/Data/Mel_Spectrograms/" --n_mels 1024 --hop_length 512 --n_fft 128 --category "Sex"

 - csv_file: this is the csv file found in the Data folder
 - csv_path: the path to the csv file
 - audio_path: the path to the AudioWAV folder downloaded from the CREMA-D github
 - image_path: the path to where you want Mel Spectrograms to be saved
 - n_mels, hop_length, n_fft = best parameters for the category audio files
 - category: either sex, age, race or emotion

The output will be 3 csv files for the train, validation and test set. 

### Generating Mel Spectrograms

3. To run generate_mel_spectrograms.py, 










The model code was referenced from the following github repository https://github.com/m3hrdadfi/soxan

The dataset used contains metadata about each audio file:
![image](https://user-images.githubusercontent.com/54903276/152839639-2366c610-afdc-41cb-92a3-c21fef91c929.png)

The value counts of the emotions to look at the class distribution:

![image](https://user-images.githubusercontent.com/54903276/152840702-e469632d-4b65-4992-8d71-4a2fcbff199a.png)




