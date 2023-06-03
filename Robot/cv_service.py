import sys

sys.path.insert(0, '../CV/InternImage/detection')
sys.path.insert(1, '../CV/SOLIDER-REID')

from typing import List
import logging

import cv2
from mmdet.apis import inference_detector, init_detector
import mmcv_custom
import mmdet_custom
import torch
import numpy as np
from model.make_model import make_model
from config import cfg
from utils.metrics import Postprocessor
from tilsdk.cv import DetectedObject

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
        self.ODConfidenceThreshold = 0.99
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
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))
        # normalize with mean and std supplied in cfg
        img = img / 255.0
        for channel in range(3):
            img[channel] -= cfg.INPUT.PIXEL_MEAN[channel]
            img[channel] /= cfg.INPUT.PIXEL_STD[channel]
        return img.astype(np.float32)

    def predict(self, suspect: List[np.ndarray], image_path: str):  # TODO: consider saving the visualization of the detections and query images
        result = inference_detector(self.ODModel, image_path)[0][0]
        boxes = result[result[:, 4] > self.ODConfidenceThreshold]
        # query is the suspect
        query = [self.load_img(q) for q in suspect]
        # gallery is cropped out boxes
        gallery = []
        for box in boxes:
            x1, y1, x2, y2, conf = box.astype(np.int32)
            gallery.append(image[y1:y2, x1:x2])

        inputs = query + [self.load_img(img) for img in gallery]
        inputs = np.stack(inputs, axis=0)  # stack the query and gallery images as batch dim

        features = self.REID(torch.from_numpy(inputs).to('cuda'))[0]

        self.REID_postprocessor.update(features.detach())  # postprocessor expects Torch tensor as it uses torch to compute stuff
        dist_mat = self.REID_postprocessor.compute()
        self.REID_postprocessor.reset()  # reset the postprocessor for next query
        results = []
        candidates = []
        for i, dist in enumerate(dist_mat):
            if dist < self.REIDThreshold:
                candidates.append((boxes[i], dist))
            else:
                results.append((boxes[i], 0))  # exceeded loose threshold already, not suspect
        if candidates:
            # get the candidate with the lowest distance
            candidates.sort(key=lambda x: x[1])
            results.append((candidates[0][0], 1))
            # append the rest as not suspect
            for i in range(1, len(candidates)):
                results.append((candidates[i][0], 0))
        return results

    def targets_from_image(self, img) -> List[DetectedObject]:  # TODO: update when more details are out. See InterImage-L with REID pipeline.ipynb
        '''Process image and return targets.
        
        Parameters
        ----------
        img : np.ndarray
            Input image.
        
        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        # h, w, c = img.shape
        # img = img[int(h * 3 / 10):int(h * 9 / 10), int(w * 1 / 10):int(w * 9 / 10), :]  # zoom 30% crop on top, 10% on bottom, 10% on left and right
        #
        # result = inference_detector(self.model, img)
        # detections = []
        #
        # current_detection_id = 1
        # for class_id, this_class_detections in enumerate(result):
        #     for detection in this_class_detections:
        #         x1, y1, x2, y2, _confidence = [float(x) for x in detection]
        #         if _confidence > 0.4:
        #             detections.append(DetectedObject(
        #                 id=current_detection_id,
        #                 cls=1 - class_id,
        #                 bbox=BoundingBox(x=(x1 + x2) / 2 + w / 10, y=(y1 + y2) / 2 + h * 3 / 10, w=x2 - x1, h=y2 - y1),
        #             ))
        #             print(f'Detected {"fallen" if class_id == 0 else "standing"}, conf {_confidence}')
        #             current_detection_id += 1
        #
        # return detections
        # DSTA will likely give raw bytes for suspect image so the data loading code is split out here and will be replaced once the details of finals come out. Same for the image. Also not sure if DSTA give images in RGB or BGR. RT-DETR expects BGR but REID expects RGB.
        query_paths = ["../CV/RT-DETR/dataset/reid/test_old/query/image_0000.png"]
        query = [cv2.imread(q) for q in query_paths]
        query = [cv2.cvtColor(q, cv2.COLOR_BGR2RGB) for q in query]
        results = self.predict(suspect=query, image_path="soccer.jpg")  # TODO: change to saving the image temporarily and passing the path, since MMCV wants path as input (np array is supported but not sure if preprocessing works so I do not want to risk)
        logger.info(results)
        return results
