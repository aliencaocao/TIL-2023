import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['LTP_PATH'] = '../ASR'  # local server for language tool

import sys

sys.path.insert(0, '../')  # for importing TensorRT_Inference

import io
from typing import Iterable
import logging

from TensorRT_Inference import TRTInference

import torchaudio
from tilsdk.localization.types import *
from transformers import Wav2Vec2Processor
import language_tool_python

logger = logging.getLogger('NLPService')


class NLPService:
    def __init__(self, preprocessor_dir: str = 'wav2vec2-conformer', model_dir: str = 'wav2vec2-conformer.trt'):
        '''
        Parameters
        ----------
        preprocessor_dir : str
            Path of preprocessor folder.
        model_dir : str
            Path of model weights.
        '''
        logger.info('Initializing NLP service...')
        logger.debug(f'Loading Wav2Vec2Processor from {preprocessor_dir}...')
        self.processor = Wav2Vec2Processor.from_pretrained(preprocessor_dir)
        logger.debug(f'Loading TensorRT engine from {model_dir}...')
        self.model = TRTInference(model_dir, verbose=True)
        logger.debug('Warming up TensorRT engine...')
        self.model.warmup({'input': np.random.randn(1, 16000)})
        logger.debug('Starting language tool...')
        self.language_tool = language_tool_python.LanguageTool('en-US', config={'cacheSize': 10000, 'pipelineCaching': True, 'maxCheckThreads': 30, 'warmUp': True})
        logger.info('NLP service initialized.')

    @staticmethod
    def clean(annotation):
        if "'" in annotation:
            logger.warning(annotation, f'has \' in it, removing')
            annotation = annotation.split("'")[0] + annotation.split("'")[1][1:]  # Tokenizer includes "'" but TIL dataset does not, remove the S following '
        annotation = ' '.join(annotation.split())  # Remove extra spaces
        return annotation

    def fix_grammar(self, annotation):
        match = self.language_tool.check(annotation)
        if match:
            for i, m in enumerate(match):
                if match[i].message == 'Possible spelling mistake found.' and match[i].replacements:
                    match[i].replacements[0] = match[i].replacements[0].split()[0]  # prevent it from adding new words
            annotation = language_tool_python.utils.correct(annotation, match).upper()
        return annotation

    def predict(self, wav: bytes) -> str:
        try:
            audio, sr = torchaudio.load(io.BytesIO(wav))
            audio = self.processor(audio, sampling_rate=16000).input_values[0][0]
            audio = np.expand_dims(audio, axis=0)
            output = self.model({'input': audio})['output']
            output = self.clean(self.processor.batch_decode(np.argmax(output, axis=-1))[0])
            output = self.fix_grammar(output)
            logger.info(f'Predicted: {output}')
            return output
        except Exception as e:
            logger.error(f'Error while predicting: {e}')
            return ''

    def locations_from_clues(self, clues: Iterable[Clue]):  # TODO: Update when finals details out
        '''Process clues and get locations of interest.
        
        Parameters
        ----------
        clues
            Clues to process.

        Returns
        -------
        lois
            Locations of interest.
        '''
        for clue in clues:
            text = self.predict(clue.audio)
        locations = [c.location for c in clues]

        return locations, locations
