import sys

sys.path.insert(0, '../CV/InternImage/detection')
sys.path.insert(1, '../CV/SOLIDER-REID')

import os
from typing import List
import logging
import time

import cv2
from mmdet.apis import inference_detector, init_detector
import mmcv_custom
import mmdet_custom
import torch
import numpy as np
from model.make_model import make_model
from config import cfg
from utils.metrics import Postprocessor

logger = logging.getLogger('CVService')
assert mmdet_custom
assert mmcv_custom


class CVService:
    def __init__(self, config: str, model: str, reid_model: str, reid_config: str):
        '''
        Parameters
        ----------
        config : str
            Path to mmdet config file.
        model : str
            Path to model checkpoint.
        reid_model : str
            Path to REID model checkpoint.
        reid_config : str
            Path to REID model yml config.
        '''
        logger.info('Initializing CV service...')
        logger.debug(f'Loading object detection model from {model} using config {config}')
        self.ODModel = init_detector(config, model, device="cuda:0")
        self.ODConfidenceThreshold = 0.9  # more lenient than 0.99 used in quals as we are taking one with highest conf later
        self.REIDThreshold = 0.0  # TODO: test on zindi using euclidean distance

        logger.debug(f'Loading SOLIDER-REID model from {reid_model} using config {reid_config}')
        cfg.merge_from_file(reid_config)
        self.REID = make_model(cfg, num_class=2, camera_num=1, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        self.REID.classifier = torch.nn.Identity()  # remove the classifier layer
        self.REID.load_param(reid_model)
        self.REID.to('cuda')
        self.REID.eval()
        self.REID_postprocessor = Postprocessor(num_query=1, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=False)  # in finals cannot use RR as threshold will be changed based on gallery size
        logger.info('CV service initialized.')

    @staticmethod
    def load_img(img: np.ndarray):  # for REID only
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        # normalize with mean and std supplied in cfg
        img = img / 255.0
        for channel in range(3):
            img[channel] -= cfg.INPUT.PIXEL_MEAN[channel]
            img[channel] /= cfg.INPUT.PIXEL_STD[channel]
        return img.astype(np.float32)

    def predict(self, suspect: List[np.ndarray], image: np.ndarray) -> tuple[np.ndarray, str]:
        """Returns image drawn with bbox and class “suspect”/”hostage”/"none". Assume image only contains 1 gallery plushie"""
        assert len(suspect) == 2, f'Expecting 2 suspects, got {len(suspect)}'
        result = inference_detector(self.ODModel, image)[0][0]
        boxes = result[result[:, 4] > self.ODConfidenceThreshold]
        if len(boxes) == 0:
            logger.info('Predicted None due to no bbox')
            return image, "none"  # no box at all, return none
        box = boxes[boxes[:, 4].argsort()][-1]  # take the highest confidence box
        query = [self.load_img(q) for q in suspect]  # 2, first is suspect, second is hostage
        x1, y1, x2, y2 = box[:4].astype(np.int32)  # throw away the confidence
        gallery = image[y1:y2, x1:x2]

        inputs = query + [self.load_img(gallery)]
        inputs = np.stack(inputs, axis=0)  # stack the query and gallery images as batch dim

        features = self.REID(torch.from_numpy(inputs).to('cuda'))[0]
        self.REID_postprocessor.update(features.detach())  # postprocessor expects Torch tensor as it uses torch to compute stuff
        dist_mat = self.REID_postprocessor.compute()  # (2, 1) array, 2 queries, 1 gallery image
        dist_mat = np.squeeze(dist_mat)  # (2,)
        logger.debug(f'Distance matrix: {dist_mat}')
        self.REID_postprocessor.reset()  # reset the postprocessor for next query

        # take the lowest distance one
        min_id = np.argmin(dist_mat)
        if dist_mat[min_id] > self.REIDThreshold:  # the min one still fail threshold so return none
            logger.info('Predicted None due to threshold')
            return image, "none"
        img_with_bbox = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        os.makedirs('CV_output', exist_ok=True)
        cv2.imwrite(f"CV_output/{time.time()}.png", img_with_bbox)
        pred = "suspect" if min_id == 0 else "hostage"
        logger.info(f'Predicted {pred}')
        return img_with_bbox, pred
