Emotion Classification for this project will consist of interpreting audio files in which sentences are read in an angry, happy, sad, neutral, disgusted or fear. The notebook in the code folder is written in Python and uses a transformer called Wav2Vec2 with a classification head to train a dataset. The dataset that is being used in this model comes from a Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D). The data contains  age, sex, race, ethnicity information from 91 actors, 48 male and 42 female actors who speak 12 sentences in 6 emotions at 4 emotion levels.

The model code was referenced from the following github repository https://github.com/m3hrdadfi/soxan

The dataset used contains metadata about each audio file:
![image](https://user-images.githubusercontent.com/54903276/152839639-2366c610-afdc-41cb-92a3-c21fef91c929.png)

The value counts of the emotions to look at the class distribution:

![image](https://user-images.githubusercontent.com/54903276/152840702-e469632d-4b65-4992-8d71-4a2fcbff199a.png)


