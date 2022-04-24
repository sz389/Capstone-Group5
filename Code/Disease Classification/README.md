# Disease Classification 

## Directory  
The files are organized like this  
code/  
26-29_09_2017_KCL/  

## Preparing data  
wget https://zenodo.org/record/2867216/files/26_29_09_2017_KCL.zip  
unzip 26_29_09_2017_KCL.zip

install packages  
pip install torchaudio  
pip install librosa  

## Code
encoder_decoder.py is used to run the autoencoder and save the encoder part of the model.  
CNN_encoder.py is used to load from the encoder and run the model on Parkinson Classification.  
KCL_tocsv.py is used to create a csv file to load the data.  
GenerateSpectrogram.py is used to generate mel-spectrograms and save them.
