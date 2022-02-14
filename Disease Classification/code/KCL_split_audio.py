# %% --------------------------------------- Imports ------------------------------------
from pydub import AudioSegment
from pydub.utils import make_chunks
import pandas as pd
import numpy as np
import os
#%%
# os.chdir('..')
# save_path = os.getcwd()+'/26-29_09_2017_KCL/test_split/'
#
# myaudio = AudioSegment.from_file("/home/ubuntu/Capstone/26-29_09_2017_KCL/SpontaneousDialogue/PD/ID34_pd_2_0_0.wav" , "wav")
# chunk_length_ms = 10000 # pydub calculates in millisec
# chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
#
# #Export all of the individual chunks as wav files
#
# for i, chunk in enumerate(chunks):
#     chunk_name = "chunk{0}.wav".format(i)
#     chunk_name = save_path+chunk_name
#     print ("exporting", chunk_name)
#     chunk.export(chunk_name, format="wav")
#%%
os.chdir('..')
os.getcwd()
audio_path = os.getcwd()+'/26-29_09_2017_KCL/'
df = pd.read_csv(audio_path+'Parkinson.csv')
file_name_list = []
for path in df['path']:
    file_name_list.append(str(path).split('/')[-1])
df['File Name'] = file_name_list
save_path = os.getcwd()+'/26-29_09_2017_KCL/KCL_split/'
print(df.head())
#%%
def split_audio(audio_path_list,file_name_list,KCL_label_list,save_path,chunk_length_ms):
    path_list = []
    label_list = []
    j=0
    for path,filename in zip(audio_path_list,file_name_list):
        myaudio = AudioSegment.from_file(path,'wav')
        chunks = make_chunks(myaudio,chunk_length_ms)
        for i, chunk in enumerate(chunks):
            chunk_name = save_path+filename.split('.')[-2]+'_chunk'+str(i)+'.wav'
            path_list.append(chunk_name)
            label_list.append(KCL_label_list[j])
            chunk.export(chunk_name, format="wav")
        j+=1
        print ("exporting", chunk_name)
    return path_list,label_list
#%%
path_list, label_list = split_audio(df['path'],df['File Name'],df['Disease'],save_path,10000)
#%%
df_csv = pd.DataFrame()
df_csv['Disease'] = label_list
df_csv['path'] = path_list
print(df_csv.head())
#%%
df_csv.to_csv(audio_path+'KCL_split.csv',index=False)