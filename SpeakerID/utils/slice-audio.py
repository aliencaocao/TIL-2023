# Slices audio into disjoint segments of specified length (in seconds)

import pandas as pd
import torchaudio
from glob import glob

def slice_audio(audio_path, segment_length_seconds):
  wav, sr = torchaudio.load(audio_path)

segment_length_seconds = 5.

gt_csv = pd.read_csv("m2d/evar/evar/metadata/til.csv")


for audio_path in glob("../m2d/evar/work/16k/til/*.wav"):
  print(audio_path)
