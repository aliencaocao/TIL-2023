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

    def predict(self, suspect: List[np.ndarray], image: np.ndarray) -> Tuple[np.ndarray, str]:
        return image, 'none'


class ASRService:
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

    def predict(self, audio_paths: List[str]) -> Optional[Tuple[int]]:
        pass
