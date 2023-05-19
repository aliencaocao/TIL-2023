import os
import glob
from zipfile import ZipFile
from string import ascii_uppercase
import pydub
import pandas as pd
import tqdm
import textgrids

def clean(annotation):
    annotation = annotation.replace("'", '').upper()  # removes ' as TIL data does not include it, change to all upper case as Wav2Vec2 tokenizer requires
    for c in annotation:  # replace all non ascii uppercase characters per TIL dataset
        if c not in ascii_uppercase + ' ':
            annotation = annotation.replace(c, '')
    return annotation.strip()

def IMDA_part12_to_TIL_single(args):
    script, part = args
    dataset = []
    PART_PATH = f'/mnt/d/IMDA/PART{part}/DATA/CHANNEL0'
    WAVE_PATH = os.path.join(PART_PATH, 'WAVE')

    speaker = os.path.basename(script).split('.')[0][1:-1]
    session = os.path.basename(script).split('.')[0][-1]
    speaker_prefix = f'SPEAKER{speaker}'
    zip_file = speaker_prefix + '.zip'
    wav_root = os.path.join(speaker_prefix, f'SESSION{session}')
    with open(script, 'r') as f:
        lines = f.readlines()
        processed_lines = []
        for i in range(0, len(lines), 2):
            processed_lines.append(lines[i].split('\t')[0] + ' ' + lines[i+1].replace('\t', ' ').strip())
        processed_lines = [l.replace(chr(0xfeff), '').strip() for l in processed_lines]  # remove \ufeff which appears at the start of txt
    with ZipFile(os.path.join(WAVE_PATH, zip_file), 'r') as zip:
        for line in processed_lines:
            wav_fname = line.split(' ')[0] + '.WAV'
            data = zip.read(os.path.join(wav_root, wav_fname).replace('\\', '/'))
            audio = pydub.AudioSegment(data)
            annotation = ' '.join(line.split(' ')[1:])
            annotation = clean(annotation)
            if len(annotation.split()) >= 2 and 2000 * 2 < len(audio) < 10000 * 10:  # ignore all audio longer than 10 seconds and shorter than 2 second and single word audio
                os.makedirs(f'/mnt/d/IMDA_TIL/audio_part{part}', exist_ok=True)
                target_path = os.path.join(f'/mnt/d/IMDA_TIL/audio_part{part}', wav_fname).replace('\\', '/')
                if not os.path.isfile(target_path):
                    audio.export(target_path, format='wav')
                dataset.append({'path': target_path, 'annotation': annotation})
    return dataset


def TextGridToTIL_single(args):
    wav_path, script, part = args
    dataset = []
    base_name = os.path.basename(script).split('.')[0]
    try:
        textgrid = textgrids.TextGrid(script)
    except UnicodeDecodeError:
        return dataset
    for _ in textgrid.keys():  # just get the first key since there is only 1 all the time but name changes
        textgrid = textgrid[_]
        break
    timeline = []
    for interval in textgrid:
        if interval.text != '<Z>':
            text = interval.text
            text = clean(text)
            if len(text.split()) >= 2 and 2 < interval.xmax - interval.xmin < 10:  # ignore all audio longer than 10 seconds and shorter than 2 second and single word audio
                timeline.append({'start': interval.xmin, 'end': interval.xmax, 'text': text})
    if not os.path.isfile(wav_path):
        if part == 6:
            return dataset  # part 6 folder structure means not all audio files can be found within the same wav folder that this function is being called with
        else:
            raise FileNotFoundError(f'{wav_path} not found')
    os.makedirs(f'/mnt/d/IMDA_TIL/audio_part{part}', exist_ok=True)
    audio = pydub.AudioSegment.from_wav(wav_path)
    i = 0
    for interval in timeline:
        start = int(interval['start'] * 1000)
        end = int(interval['end'] * 1000)
        annotation = interval['text']
        target_path = os.path.join(f'/mnt/d/IMDA_TIL/audio_part{part}', base_name + str(i) + '.wav').replace('\\', '/')
        if not os.path.isfile(target_path):
            audio[start:end].export(target_path, format='wav')
        dataset.append({'path': target_path, 'annotation': annotation})
        i += 1
    return dataset