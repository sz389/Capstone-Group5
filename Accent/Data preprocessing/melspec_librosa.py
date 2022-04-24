
# This file is used to generate Mel spectrograms.
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio

sample_rate = 16000
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128
import librosa.feature
def melspec_librosa(x, sample_rate = 16000,n_fft = 1024,win_length = None,hop_length = 512,n_mels = 128):
    return librosa.feature.melspectrogram(
    y = x,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    n_mels=n_mels,
    norm='slaney',
    htk=True,
)
def save_spectrogram(spec,file, path, aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  axs.get_xaxis().set_visible(False)
  axs.get_yaxis().set_visible(False)
  plt.axis('off')
  # plt.show(block=False)
  plt.savefig(
      path + '/' + file + ".jpg",
      # augmented_images_path + "Augmented_" + file + "_" + str(
      #     augmentation_times) + ".jpg",
      bbox_inches='tight', pad_inches=0)

  return path + '/' + file + ".jpg"
