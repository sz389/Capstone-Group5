The wav files are found in this repository under: Speech_Recognition/Data/Emotion_Age_Race_Sex/

To generate the spectrograms needed for the other code files, you must run Mel_Spectrograms.py using CREMA_D_nopath.csv and the data. 

To run autoencoder, you must have CREMA_D_Image.csv and CREMA_D_nopath.csv as well as the mel spectrograms generated previously. 

To run Data Augmentation.py, you only need CREMA_D_nopath.csv as well as a folder of the mel spectrograms.
