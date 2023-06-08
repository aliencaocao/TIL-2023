import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['LTP_PATH'] = '../ASR'  # local server for language tool

import sys

sys.path.insert(0, '../')  # for importing TensorRT_Inference

import logging

from TensorRT_Inference import TRTInference

import Levenshtein
import torchaudio
from tilsdk.localization.types import *
from transformers import Wav2Vec2Processor
import language_tool_python
import numpy as np

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
            outputs = (self.clean(anno) for anno in self.processor.batch_decode(np.argmax(outputs, axis=-1)))
            outputs_text = tuple(self.fix_grammar(anno) for anno in outputs)
            logger.debug(f'Predicted texts: {outputs_text}')
            outputs = tuple(self.find_digit(anno) for anno in outputs_text)  # contains None
            assert len(outputs) == len(audio_paths)  # make sure every audio has a corresponding output, even if it has no predicted digits
            outputs_without_None = tuple(o for o in outputs if o is not None)
            logger.info(f'Predicted digits: {outputs_without_None}')
            if len(outputs_without_None) != len(audio_paths):
                # Do fuzzy retrieval of closest word that matches any of the digits using Levenshtein Distance and common word heuristics
                logger.warning(f'{len(audio_paths) - len(outputs_without_None)} audio(s) has no predicted digits! Using Levenshtein Distance to do fuzzy retrieval')
                outputs = np.array(outputs)  # convert to np array for faster index finding ops
                # noinspection PyComparisonWithNone
                None_idx = np.where(outputs == None)[0]
                missing_texts = [outputs_text[i] for i in None_idx]
                outputs_text = list(outputs_text)  # to allow item assignment below
                for idx, s in zip(None_idx, missing_texts):
                    s = s.split()
                    digits_dist = []
                    for digit in self.digits:
                        digits_dist.append([Levenshtein.distance(digit, word) for word in s])
                    digits_dist = np.array(digits_dist)
                    min_dist = np.min(np.min(digits_dist, axis=0), axis=0)
                    min_dist_idx = np.where(digits_dist == min_dist)
                    if len(min_dist_idx) == 1:  # there is only 1 word that has the lowest distance to any of the digits, just take it
                        logger.info(f'Replacing {s[min_dist_idx[0][0]]} with {self.digits[min_dist_idx[0][1]]}')
                        s[min_dist_idx[0][0]] = self.digits[min_dist_idx[0][1]]
                        outputs_text[idx] = ' '.join(s)
                        logger.info(f'Predicted text after fuzzy retrieval: {outputs_text[idx]}')
                    else:
                        logger.warning('Multiple words have the same distance to the digits. Choosing 1 based on heuristics')
                        common_words = ['TO', 'THE', 'THEY', 'HE', 'SHE', 'A', 'WE']  # if these words are the one that is closest to the digit, then we choose the next one as these are common words and unlikely to be the digit. E.g. TWO -> TO has distance of only 1 but it is most likely wrong
                        for i in range(len(min_dist_idx)):
                            if s[min_dist_idx[i][0]] in common_words:
                                continue
                            else:
                                logger.info(f'Replacing {s[min_dist_idx[i][0]]} with {self.digits[min_dist_idx[i][1]]}')
                                s[min_dist_idx[0][0]] = self.digits[min_dist_idx[0][1]]
                                outputs_text[idx] = ' '.join(s)
                                logger.info(f'Predicted text after fuzzy retrieval: {outputs_text[idx]}')
                                break
                outputs = (self.find_digit(anno) for anno in outputs_text)  # should not contain None anymore
                outputs_without_None = tuple(o for o in outputs if o is not None)
                assert len(outputs_without_None) == len(audio_paths)  # make sure every audio has a corresponding predicted digit
                logger.info(f'Predicted digits after fuzzy retrieval: {outputs_without_None}')
            return outputs_without_None
        except Exception as e:
            logger.error(f'Error while predicting: {e}')
            return None
