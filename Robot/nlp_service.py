import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['LTP_PATH'] = '../ASR'  # local server for language tool

import sys

sys.path.insert(0, '../')  # for importing TensorRT_Inference

import logging

from TensorRT_Inference import TRTInference

import torchaudio
from tilsdk.localization.types import *
from transformers import Wav2Vec2Processor
import language_tool_python

logger = logging.getLogger('NLPService')


class ASRService:
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
        self.digits = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', "NINE"]
        self.digits_to_int = {d: i for i, d in enumerate(self.digits)}
        logger.debug(f'Loading Wav2Vec2Processor from {preprocessor_dir}...')
        self.processor = Wav2Vec2Processor.from_pretrained(preprocessor_dir)
        logger.debug(f'Loading TensorRT engine from {model_dir}...')
        self.model = TRTInference(model_dir, verbose=True)
        logger.debug('Warming up TensorRT engine...')
        self.model.warmup({'input': np.random.randn(16000)})
        logger.debug('Starting language tool...')
        self.language_tool = language_tool_python.LanguageTool('en-US', config={'cacheSize': 10000, 'pipelineCaching': True, 'maxCheckThreads': 30, 'warmUp': True})
        logger.info('NLP service initialized.')

    @staticmethod
    def clean(annotation: str) -> str:
        if "'" in annotation:
            logger.warning(annotation, f'has \' in it, removing')
            annotation = annotation.split("'")[0] + annotation.split("'")[1][1:]  # Tokenizer includes "'" but TIL dataset does not, remove the S following '
        annotation = ' '.join(annotation.split())  # Remove extra spaces
        return annotation

    def fix_grammar(self, annotation: str) -> str:
        match = self.language_tool.check(annotation)
        if match:
            for i, m in enumerate(match):
                if match[i].message == 'Possible spelling mistake found.' and match[i].replacements:
                    match[i].replacements[0] = match[i].replacements[0].split()[0]  # prevent it from adding new words
            annotation = language_tool_python.utils.correct(annotation, match).upper()
        return annotation

    def find_digit(self, annotation: str) -> Optional[int]:
        for d in self.digits:
            if d in annotation:
                return self.digits_to_int[d]  # take the first one found
        return None

    def predict(self, audio_paths: list[str]) -> Optional[tuple[int]]:
        try:
            logger.info(f'Predicting {len(audio_paths)} audio(s)...')
            audio_paths.sort()
            outputs = []
            for audio in audio_paths:
                audio = torchaudio.load(audio)[0]
                audio = self.processor(audio, sampling_rate=16000).input_values[0][0]
                output = self.model({'input': audio})['output'][0]  # remove batch dimension
                outputs.append(output)
            outputs = [self.clean(anno) for anno in self.processor.batch_decode(np.argmax(outputs, axis=-1))]
            outputs = [self.fix_grammar(anno) for anno in outputs]
            logger.debug(f'Predicted texts: {outputs}')
            outputs = tuple(self.find_digit(anno) for anno in outputs)
            outputs = tuple(o for o in outputs if o is not None)
            logger.info(f'Predicted digits: {outputs}')
            if len(outputs) != len(audio_paths):
                logger.warning(f'{len(audio_paths) - len(outputs)} has no predicted digits!')  # TODO: add fuzzy retrival in this case, perhaps using Levenshtein Distance
            return outputs
        except Exception as e:
            logger.error(f'Error while predicting: {e}')
            return None
