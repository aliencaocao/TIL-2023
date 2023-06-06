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
        self.model.warmup({'input': np.random.randn(1, 16000)})
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
            audios = []
            for a in audio_paths:
                audios.append(torchaudio.load(a)[0])
            audios = self.processor(audios, sampling_rate=16000).input_values[0][0]  # batched already so no need expand dims later
            output = self.model({'input': audios})['output']
            output = [self.clean(anno) for anno in self.processor.batch_decode(np.argmax(output, axis=-1))]
            output = [self.fix_grammar(anno) for anno in output]
            logger.debug(f'Predicted texts: {output}')
            output = tuple(self.find_digit(anno) for anno in output)
            output = tuple(o for o in output if o is not None)
            logger.info(f'Predicted digits: {output}')
            if len(output) != len(audio_paths):
                logger.warning(f'{len(audio_paths) - len(output)} has no predicted digits!')  # TODO: add fuzzy retrival in this case, perhaps using Levenshtein Distance
            return output
        except Exception as e:
            logger.error(f'Error while predicting: {e}')
            return None
