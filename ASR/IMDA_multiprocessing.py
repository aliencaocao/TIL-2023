import os
import glob
import multiprocessing
import tqdm
from IMDA import IMDA_part12_to_TIL_single, TextGridToTIL_single
import pickle
import pandas as pd

pool = multiprocessing.Pool(5)

def IMDA_part12_to_TIL(part: int):
    PART_PATH = f'/mnt/d/IMDA/PART{part}/DATA/CHANNEL0'
    SCRIPT_PATH = os.path.join(PART_PATH, 'SCRIPT')

    script_files = glob.glob(os.path.join(SCRIPT_PATH, '*.TXT'))

    return list(tqdm.tqdm(pool.imap(IMDA_part12_to_TIL_single, zip(script_files, [part] * len(script_files))), total=len(script_files)))

def TextGridToTIL(wav_path, SCRIPT_PATH, part: int):
    wav_files = glob.glob(os.path.join(wav_path, '*.wav')) + glob.glob(os.path.join(wav_path, '*.WAV'))
    script_files = glob.glob(os.path.join(SCRIPT_PATH, '*.TextGrid'))
    return list(tqdm.tqdm(pool.imap(TextGridToTIL_single, zip(wav_files, script_files, [part] * len(script_files))), total=len(script_files)))


dataset = []
if os.path.isfile('dataset.pkl'):
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

try:
    # dataset_lists = IMDA_part12_to_TIL(1)
    # for d in dataset_lists:
    #     dataset.extend(d)
    # IMDA_part12_to_TIL(2)  # part 2 is full of SG names, not considered english

    # dataset_lists = TextGridToTIL('/mnt/d/IMDA/PART3/Audio Same CloseMic', '/mnt/d/IMDA/PART3/Scripts Same', 3)
    # for d in dataset_lists:
    #     dataset.extend(d)

    # dataset_lists = TextGridToTIL('/mnt/d/IMDA/PART3/Audio Same CloseMic', '/mnt/d/IMDA/PART3/Scripts Same', 3)
    # for d in dataset_lists:
    #     dataset.extend(d)


    # part 5 is a bit different audio folder structure
    def IMDA_part5_to_TIL(wav_path, script_path):
        dataset = []
        dataset_lists = []
        speaker_folders = os.listdir(wav_path)
        speaker_paths = [os.path.join(wav_path, speaker_folder) for speaker_folder in speaker_folders]
        for speaker in tqdm.tqdm(speaker_paths):
            dataset_lists += TextGridToTIL(speaker, script_path, 5)
        for d in dataset_lists:
            dataset.extend(d)
        return dataset

    # dataset.extend(IMDA_part5_to_TIL('/mnt/d/IMDA/PART5/Debate Audio', '/mnt/d/IMDA/PART5/Debate Scripts'))
    # dataset.extend(IMDA_part5_to_TIL('/mnt/d/IMDA/PART5/Finance + Emotions Audio', '/mnt/d/IMDA/PART5/Finance + Emotions Scripts'))

    # part 6 different audio folder structure
    wav_folders = glob.glob('/mnt/d/IMDA/PART6/Call Centre Design */Audio/*/*/')
    script_folders = [wav_folder.split('Audio')[0] + 'Scripts/' for wav_folder in wav_folders]
    for wav_path, script_path in tqdm.tqdm(zip(wav_folders, script_folders)):
        dataset_lists = TextGridToTIL(wav_path, script_path, 6)
        for d in dataset_lists:
            dataset.extend(d)

except:
    with open('dataset.pkl', 'wb+') as f:
        pickle.dump(dataset, f)
    df = pd.DataFrame(dataset)
    print(len(df))
    df.to_csv('/mnt/d/IMDA_TIL/IMDA_TIL.csv', index=False)
else:
    with open('dataset.pkl', 'wb+') as f:
        pickle.dump(dataset, f)
    df = pd.DataFrame(dataset)
    print(len(df))
    df.to_csv('/mnt/d/IMDA_TIL/IMDA_TIL.csv', index=False)
