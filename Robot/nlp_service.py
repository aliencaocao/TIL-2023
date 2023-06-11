import os

os.environ['LTP_PATH'] = '../ASR'  # local server for language tool

import sys
sys.path.insert(0, "../SpeakerID/m2d")
sys.path.insert(0, "../SpeakerID/m2d/evar")

import logging

import Levenshtein
from tilsdk.localization.types import *
from transformers import pipeline
import language_tool_python
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
import pickle
from pathlib import Path
from evar.ar_m2d import AR_M2D
from finetune import TaskNetwork
from lineareval import make_cfg
from evar.common import kwarg_cfg

logger = logging.getLogger('NLPService')

class SpeakerIDService:
    def __init__(self, config_path: str = '../SpeakerID/m2d/evar/config/m2d.yaml', run_dir: str, model_filename: str):
        logger.info('Initializing SpeakerID service...')

        run_dir = Path(run_dir)
        self.segment_length_seconds = 3. # TODO: change according to slice length
        
        logger.debug(f'Loading SpeakerID config from {config_path}...')
        device = torch.device("cuda")
        cfg, n_folds, activation, balanced = make_cfg(
            config_file=config_path,
            task="til",
            options="",
        )
        cfg.weight_file = "../SpeakerID/m2d/weights/msm_mae_vit_base-80x608p16x16-220924-mr75/checkpoint-300.pth"
        cfg.unit_sec = self.segment_length_seconds # TODO: change unit_sec according to slice length
        cfg.runtime_cfg = kwarg_cfg(n_class=40, hidden=()) # TODO: change n_class according to no. of unique speakers

        state_dict_path = run_dir / model_filename
        logger.debug(f"Loading model state_dict from {state_dict_path}")
        state_dict = torch.load(state_dict_path)

        norm_stats_path = run_dir / "norm_stats.pt"
        logger.debug(f"Loading normalization stats from {norm_stats_path}")
        norm_stats = torch.load(norm_stats_path)

        self.teamname = '10000SGDMRT'
        classes_pickle_path = run_dir / "classes.pkl"
        logger.debug(f"Loading class names pickle from {classes_pickle_path}")
        with open(classes_pickle_path, "rb") as classes_pickle:
            self.class_names = pickle.load(classes_pickle).astype(str)
        self.our_team_ids = np.where(np.char.startswith(self.class_names, self.teamname))[0]
        assert len(self.our_team_ids) == 4, "There should be 4 members in our team"

        logger.debug(f"Instantiating AR_M2D backbone")
        backbone = AR_M2D(cfg, inference_mode=True, norm_stats=norm_stats).to(device)

        logger.debug(f"Instantiating TaskNetwork")
        self.model = TaskNetwork(cfg, ar=backbone).to(device)
        self.model = torch.nn.DataParallel(self.model).to(device)

        logger.debug(f"Loading TaskNetwork state_dict")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # warm up
        logger.debug("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                self.model(torch.randn(3, 22050).to(device))

        logger.info("SpeakerID service successfully initialized.")
    
    def predict(self, audio_path: List[str]) -> Optional[str]:
        """Takes in a list of audio paths and return a string in format audio1_teamNameOne_member2, where the Team is the opponent"""
        # load a full-length audio from DSTA
        try:
            # assert len(audio_path) == 2, "There should be 2 audio files"
            results = {}
            logits = {}
            for path in audio_path:
                wav, sr = torchaudio.load(path)

                # get the length of each segment in samples
                segment_length_samples = int(self.segment_length_seconds * sr)

                # slice the original full-length audio into segments of length segment_length_samples
                segments = list(torch.split(wav, segment_length_samples, dim=1))

                # get the length of each segment, in samples
                segment_lengths = torch.Tensor([s.shape[1] for s in segments]).to('cuda:0')

                # pad the last segment to the same length as the rest
                # the last segment will be the one that's not long enough
                segments[-1] = F.pad(segments[-1], (0, segment_length_samples - segments[-1].shape[1]), value=0)

                # concat all the segments into a batch and do the forward pass
                segments_batch = torch.cat(segments).to('cuda:0')
                with torch.no_grad():
                    model_output = self.model(segments_batch)

                # multiply by length of each wav to get weighted average
                model_output = model_output * segment_lengths.unsqueeze(-1)
                logits_averaged = torch.mean(model_output, dim=0)
                pred_idx = torch.argmax(logits_averaged)
                pred = self.class_names[pred_idx]
                results[path.split(os.sep)[-1][:-4]] = pred
                logits[path.split(os.sep)[-1][:-4]] = logits_averaged.cpu().numpy()
                logger.info(f'Predicted: {results}')

            # Post-processing to ensure exactly one not-our-team speaker is predicted
            our_team_counts = len([i for i in results.values() if i.startswith(self.teamname)])
            if our_team_counts == 1:
                result = [k + '_' + v for k, v in results.items() if not k.startswith(self.teamname)][0]  # audio1_teamNameOne_member2
            else:  # more than 1 OR no one predicted is our team, take the one with highest confidence as our team and return the other one to be the opponent team
                logger.warning(f'Predicted {our_team_counts} audios to be from {self.teamname}! Choosing the one with lowest confidence in {self.teamname} members as opponent.')
                min_confidence = 0
                for k, v in results.items():
                    min_our_team_confidence = np.min(logits[k][self.our_team_ids])
                    if min_our_team_confidence < min_confidence:
                        min_confidence = min_our_team_confidence
                        result = k
                logits_without_our_team = logits[result].copy()
                logits_without_our_team[self.our_team_ids] = -1e99  # set to very low value so that np.argmax will not choose our team
                result = result + '_' + self.class_names[np.argmax(logits_without_our_team)]
                logger.info(f'Predicted after choosing lowest conf: {result}')
            return result
        except Exception as e:
            logger.error(f'Error while predicting: {e}')
            return None


class ASRService:
    def __init__(self, model_dir: str = 'wav2vec2-conformer'):
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
        logger.debug(f'Loading model from {model_dir}...')
        self.model = pipeline("automatic-speech-recognition", model=model_dir, batch_size=4, device="cuda:0")
        for _ in range(3):  # warm up
            self.model(['data/audio/evala_00001.wav'])
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

    def predict(self, audio_paths: List[str]) -> Optional[Tuple[int]]:
        try:
            logger.info(f'Predicting {len(audio_paths)} audio(s)...')
            audio_paths.sort()
            outputs = self.model(audio_paths)
            outputs = (self.clean(i['text']) for i in outputs)
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
