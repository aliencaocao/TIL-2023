from typing import List, Any
from tilsdk.cv.types import *
from tilsdk.localization.types import *
from tilsdk.cv import DetectedObject, BoundingBox
from typing import Iterable, List


class CVService:
    '''Mock CV Service.
    
    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir: str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        # Does nothing.
        pass

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
    '''Mock NLP Service.
    
    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        pass

    def locations_from_clues(self, clues:Iterable[Clue]) -> List[RealLocation]:
        '''Process clues and get locations of interest.
        
        Mock returns location of all clues.
        '''
        locations = [c.location for c in clues]

        return locations, locations