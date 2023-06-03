import inspect

from tilsdk import *  # import the SDK
from tilsdk.utilities import PIDController  # import optional useful things

from cv_service import CVService
from nlp_service import NLPService

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

# Define config variables in an easily accessible location
# You may consider using a config file
REACHED_THRESHOLD_M = 0.3  # TODO: Participant may tune, in meters
ANGLE_THRESHOLD_DEG = 20.0  # TODO: Participant may tune.
ROBOT_RADIUS_M = 0.17  # TODO: Participant may tune. 0.390 * 0.245 (L x W)
tracker = PIDController(Kp=(0.35, 0.2), Ki=(0.1, 0.0), Kd=(0, 0))
NLP_PREPROCESSOR_DIR = 'finals_audio_model'
NLP_MODEL_DIR = 'model.onnx'
CV_CONFIG_DIR = 'vfnet.py'
CV_MODEL_DIR = 'epoch_13.pth'
prev_img_rpt_time = 0

cv_service = CVService(config_file=CV_CONFIG_DIR, checkpoint_file=CV_MODEL_DIR)
import cv2

img = cv2.imread('sample.png', 3)
print(cv_service.targets_from_image(img))
nlp_service = NLPService(preprocessor_dir=NLP_PREPROCESSOR_DIR, model_dir=NLP_MODEL_DIR)
print(nlp_service.predict(open('sample.wav', 'rb').read()))

import numpy as np
import pyastar2d

print(inspect.getfile(pyastar2d))
weights = np.array([[1, 3, 3, 3, 3],
                    [2, 1, 3, 3, 3],
                    [2, 2, 1, 3, 3],
                    [2, 2, 2, 1, 3],
                    [2, 2, 2, 2, 1]], dtype=np.float32)
path = pyastar2d.astar_path(weights, (0, 0), (4, 4), allow_diagonal=True)
print(path)
input()
