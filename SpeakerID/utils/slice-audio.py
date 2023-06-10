# Slices audio into disjoint segments of specified length (in seconds)

import pandas as pd
import torchaudio
from pathlib import Path
import os

########################## SETTINGS ##########################
segment_length_seconds = 2.
input_csv_path = Path("../m2d/evar/evar/metadata/til.csv")
sliced_csv_path = Path("../m2d/evar/evar/metadata/til_sliced.csv")
input_audio_dir = Path("../m2d/evar/work/16k/til")
sliced_output_dir = Path("../m2d/evar/work/16k/til_sliced")

os.makedirs(sliced_output_dir, exist_ok=True)

def slice_audio(audio_path, segment_length_seconds):
  wav, sr = torchaudio.load(audio_path)
  segment_length_samples = int(segment_length_seconds * sr)

  slices = []
  curr_start = 0
  curr_end = min(segment_length_samples, wav.shape[1])
  while True:
    if curr_start >= wav.shape[1]:
      break

    slices.append(wav[:, curr_start:curr_end])
    curr_start = curr_end
    curr_end += segment_length_samples
  
  return slices, sr

def df_map_fn(row):
  slices, sr = slice_audio(input_audio_dir / row['file_name'], segment_length_seconds)
  
  # save audio slices
  new_filenames = []
  for i, slice in enumerate(slices):
    slice_filename = sliced_output_dir / (os.path.splitext(row["file_name"])[0] + f"-slice-{i+1}.wav")
    new_filenames.append(slice_filename.name)
    torchaudio.save(slice_filename, slice, sr)

  # reassign file_name column for exploding later
  row['file_name'] = new_filenames
  return row

df = pd.read_csv(input_csv_path)
df = df.apply(df_map_fn, axis=1).explode("file_name")
df.to_csv(sliced_csv_path, index=False)
