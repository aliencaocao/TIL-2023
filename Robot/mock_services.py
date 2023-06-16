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
        return image, 'suspect'

class SpeakerIDService:
    def __init__(self, config_path: str, run_dir: str, model_filename: str, FRCRN_path: str, DeepFilterNet3_path: str, current_opponent: str):
        pass

    def predict(self, audio_path: str) -> str:
        return 0

class ASRService:
    '''
    Parameters
    ----------
    preprocessor_dir : str
        Path of preprocessor folder.
    model_dir : str
        Path of model weights.
    '''

    def __init__(self, model_dir: str = 'wav2vec2-conformer'):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        pass

    def predict(self, audio_paths: List[str]) -> Optional[Tuple[int]]:
        return (9,)
