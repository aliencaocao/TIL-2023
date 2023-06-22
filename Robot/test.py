# Script to test all libraries and dependencies are installed correctly
import os
os.environ['PATH'] += ':.'  # add current dir to PATH for ffmpeg
# only works in linux
assert os.name == 'posix', 'This script only works in linux'
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')
formatter = logging.Formatter(fmt='[%(levelname)s][%(asctime)s][%(name)s]: %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
loggers = [logging.getLogger(name) for name in [__name__, 'NLPService', 'CVService', 'Navigation']]
logger = loggers[0]  # the main one
NavLogger = loggers[3]  # the navigation one
logger.name = 'Main'
for l in loggers:
    l.propagate = False
    l.addHandler(stream_handler)
    l.setLevel(logging.DEBUG)

# planner related
logger.info('Testing planner...')
import pyastar2d
import numpy as np
weights = np.array([[1, 3, 3, 3, 3],
                    [2, 1, 3, 3, 3],
                    [2, 2, 1, 3, 3],
                    [2, 2, 2, 1, 3],
                    [2, 2, 2, 2, 1]], dtype=np.float32)
assert (pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=True) == np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])).all()

# NLP services
logger.info('Testing NLP...')
from nlp_service import SpeakerIDService, ASRService

ASR_MODEL_DIR = '../ASR/wav2vec2-conformer'
SPEAKERID_RUN_DIR = '../SpeakerID/m2d/evar/logs/til_ar_m2d.AR_M2D_cb0a37cc'
SPEAKERID_MODEL_FILENAME = 'weights_ep866it1-0.90000_loss0.0160.pth' # this is a FILENAME, not a full path
SPEAKERID_CONFIG_PATH = '../SpeakerID/m2d/evar/config/m2d.yaml'
FRCRN_path = '../SpeakerID/speech_frcrn_ans_cirm_16k'
DeepFilterNet3_path = '../SpeakerID/DeepFilterNet3/'
current_opponent = 'HUGGINGROBOT'

asr_service = ASRService(ASR_MODEL_DIR)
speakerid_service = SpeakerIDService(SPEAKERID_CONFIG_PATH, SPEAKERID_RUN_DIR, SPEAKERID_MODEL_FILENAME, FRCRN_path, DeepFilterNet3_path, current_opponent)
logger.info('Predicting ASR...')
r = asr_service.predict(['data/audio/evala_00001.wav'])
assert r == (9,), r
logger.info('Predicting SpeakerID...')
r = speakerid_service.predict(['data/audio/audio1.wav', 'data/audio/audio2.wav'])
assert r == 'audio2_HUGGINGROBOT_memberB', r
asr_service.language_tool.close()

# CV service
logger.info('Testing CV...')
import cv2
from cv_service import CVService
OD_CONFIG_PATH = '../CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/cascade_internimage_l_fpn_3x_coco_custom.py'
OD_MODEL_PATH = '../CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/InternImage-L epoch_12 stripped.pth'
REID_MODEL_PATH = '../CV/SOLIDER-REID/log_SGD_500epoch_continue_1e-4LR_expanded/transformer_21_map0.941492492396344_acc0.8535950183868408.pth'
REID_CONFIG_PATH = '../CV/SOLIDER-REID/TIL.yml'
suspect_img = cv2.imread('data/imgs/SUSPECT_4.jpg')
hostage_img = cv2.imread('data/imgs/HOSTAGE.jpg')
cv_service = CVService(OD_CONFIG_PATH, OD_MODEL_PATH, REID_MODEL_PATH, REID_CONFIG_PATH)
img = cv2.imread('data/imgs/image_0000.png')
answer = cv_service.predict([suspect_img, hostage_img], img, 100, 280, 100, 100)[1]
assert answer == 'suspect', answer

logger.info('All tests passed!')