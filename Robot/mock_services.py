from typing import Iterable, List

from tilsdk.cv import BoundingBox, DetectedObject
from tilsdk.localization.types import *


class CVService:
    '''Mock CV Service.
    
    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

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

    def targets_from_image(self, img: Any) -> List[DetectedObject]:
        '''Process image and return targets.
        
        Parameters
        ----------
        img : Any
            Input image.
        
        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        # dummy data
        bbox = BoundingBox(100, 100, 300, 50)
        obj = DetectedObject("1", "fallen", bbox)
        return [obj]


class NLPService:
    '''
    Parameters
    ----------
    preprocessor_dir : str
        Path of preprocessor folder.
    model_dir : str
        Path of model weights.
    '''

    def __init__(self, preprocessor_dir: str = 'wav2vec2-conformer', model_dir: str = 'wav2vec2-conformer.trt'):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        pass

    def locations_from_clues(self, clues) -> List[RealLocation]:
        '''Process clues and get locations of interest.
        
        Mock returns location of all clues.
        '''
        locations = [c.location for c in clues]

        return locations, locations
