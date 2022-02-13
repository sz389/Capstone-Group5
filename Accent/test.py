# %% --------------------------------------- Reading in CSV File with Labels -------------------------------------------------------------------
import pandas as pd
csv_path = '/home/ubuntu/Capstone/'
df = pd.read_csv(csv_path + 'New Accent_Archive_Clean.csv', index_col = 0)
df = df[['filename', 'path', 'native_language']]
# only use top 10
top10 = ['english','spanish','arabic','mandarin','french',
         'korean','portuguese','russian','dutch','turkish']
df1 = df[df['native_language'].isin(top10)]
df = df1
#df = df.drop('path', 1)
#df['path'] = '/home/ubuntu/Capstone/recordings/recordings/'+ df['filename'] + '.mp3'
#df['image_name'] = df['filename'] + ".jpg"
df.head()
#df.to_csv("/home/ubuntu/Capstone/Accent_Image.csv")

from pydub import AudioSegment
from pydub.utils import make_chunks
import re

test_path = '/home/ubuntu/Capstone/recordings/recordings/'
mp3_path = '/home/ubuntu/Capstone/Splitmp3/'
# use csv to get file names
files  = df['filename']
df = pd.DataFrame()

for filename in files:
  myaudio = AudioSegment.from_file(test_path + filename + ".mp3")
  chunk_length_ms = 5000 # pydub calculates in millisec
  chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

  for i, chunk in enumerate(chunks):
      chunk_name =filename+'_'+ str(i) +".mp3"#.format(i)
      print("exporting", chunk_name)
      chunk.export(mp3_path+chunk_name, format="mp3")
      native_language = re.split('(\d+)', chunk_name)
      df = df.append({'filename': chunk_name,
                      'path': mp3_path + chunk_name,
                      'native_language': native_language[0]},
                     ignore_index=True)
  df.to_csv(csv_path + 'splitmp3.csv')





