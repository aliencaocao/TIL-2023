import sys

from tilsdk.localization.types import *

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
sys.path.pop(0)
sys.path.pop(0)


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
        self.ODConfidenceThreshold = 0.9 # more lenient than 0.99 used in quals as we are taking one with highest conf later
        self.REIDThreshold = 1.0

        logger.debug(f'Loading SOLIDER-REID model from {reid_model} using config {reid_config}')
        cfg.merge_from_file(reid_config)
        self.REID = make_model(cfg, num_class=2, camera_num=1, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        self.REID.classifier = torch.nn.Identity()  # remove the classifier layer
        self.REID.load_param(reid_model)
        self.REID.to('cuda')
        self.REID.eval()
        self.REID_postprocessor = Postprocessor(num_query=2, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=False)  # in finals cannot use RR as threshold will be changed based on gallery size, num_query is 2 as we have 1 suspect and 1 hostage
        os.makedirs('CV_output', exist_ok=True)
        logger.info('CV service initialized.')

    @staticmethod
    def load_img(img: np.ndarray):  # for REID only
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        # normalize with mean and std supplied in cfg
        img = img / 255.0
        for channel in range(3):
            img[channel] -= cfg.INPUT.PIXEL_MEAN[channel]
            img[channel] /= cfg.INPUT.PIXEL_STD[channel]
        return img.astype(np.float32)

    def predict(self, 
                suspect: List[np.ndarray], 
                image_orig: np.ndarray, 
                crop_from_top_pixels: int, 
                crop_from_bottom_pixels: int,
                crop_from_left_pixels: int,
                crop_from_right_pixels: int,
        ) -> Tuple[np.ndarray, str]:
        """Returns image drawn with bbox and class “suspect”/”hostage”/"none". Assume image only contains 1 gallery plushie"""
        logger.info('Predicting CV...')
        assert len(suspect) == 2, f'Expecting 2 suspects, got {len(suspect)}'

        # crop the image
        image = image_orig[crop_from_top_pixels:-crop_from_bottom_pixels,
                           crop_from_left_pixels:-crop_from_right_pixels,
                           :]

        result = inference_detector(self.ODModel, image)[0][0]
        logger.debug(f"OD bbox result: {result}")
        boxes = result[result[:, 4] > self.ODConfidenceThreshold]
        if len(boxes) == 0:
            logger.info('Predicted None due to no bbox')
            if not cv2.imwrite(f"CV_output/{int(time.time())}.png", image_orig):
                logger.warning('Failed to save image')
            return image_orig, "none"  # no box at all, return none
        query = [self.load_img(q) for q in suspect]  # 2, first is suspect, second is hostage
        gallery = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4].astype(np.int32)  # throw away the confidence
            gallery.append(image[y1:y2, x1:x2])

        inputs = query + [self.load_img(pic) for pic in gallery]
        inputs = np.stack(inputs, axis=0)  # stack the query and gallery images as batch dim

        features = self.REID(torch.from_numpy(inputs).to('cuda'))[0]
        self.REID_postprocessor.update(features.detach())  # postprocessor expects Torch tensor as it uses torch to compute stuff
        dist_mat = self.REID_postprocessor.compute()  # (2, x) array, 2 queries, x gallery image
        logger.debug(f'Distance matrix: {dist_mat}')
        self.REID_postprocessor.reset()  # reset the postprocessor for next query

        # take the lowest distance one
        min_id = np.unravel_index(dist_mat.argmin(), dist_mat.shape)
        if dist_mat[min_id] > self.REIDThreshold:  # the min one still fail threshold so return none
            logger.info('Predicted None due to threshold')
            if not cv2.imwrite(f"CV_output/{int(time.time())}.png", image_orig):
                logger.warning('Failed to save image')
            return image_orig, "none"
        # take the class and gallery image with min dist
        pred = "suspect" if min_id[0] == 0 else "hostage"
        box = boxes[min_id[1]]  # choose the corresponding box that has the min dist
        x1, y1, x2, y2 = box[:4].astype(np.int32)
        # correct for cropping
        y1 += crop_from_top_pixels
        y2 += crop_from_top_pixels
        x1 += crop_from_left_pixels
        x2 += crop_from_left_pixels
        img_with_bbox = cv2.rectangle(image_orig, (x1, y1), (x2, y2), (0, 255 if pred == 'hostage' else 0, 255 if pred == 'suspect' else 0), 2)

        if not cv2.imwrite(f"CV_output/{int(time.time())}.png", img_with_bbox):
            logger.warning('Failed to save image')
        logger.info(f'Predicted {pred}')
        return img_with_bbox, pred