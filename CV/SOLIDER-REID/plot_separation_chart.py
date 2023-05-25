import argparse
import os

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

# This file is used to plot the separation chart to find an optimal threshold for ReID.
# It is largely similar to infer.py.
# However, instead of inferring on the test set, we infer on the train + val set.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReID Inference')
    parser.add_argument('--config_file', default="", help='Path to YAML config file.', type=str)
    parser.add_argument('opts',
                        help='Modify config options using the command line.',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.INFERENCE_MODE = False
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger('reid', output_dir, if_train=False)
    logger.info(args)

    if args.config_file != '':
        logger.info('Loaded configuration file {}'.format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    logger.info('Running with config:\n{}'.format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    inference_loader = make_dataloader(cfg)
    model = make_model(
        cfg,
        num_class=2, # because we have two classes: non-suspect (0) and suspect (1).
        camera_num=1, # because we instantiate a TILCustomDataset for each test set image.
        view_num=1, # because we are working with single images, not videos. It is not applicable to us.
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT
    )
    # Semantic weight is the ratio of (Semantic Info) / (Visual Info).
    # Since ReID is a visual task, we want to weight the visual info more.
    # Thus, a semantic weight is 0.2 is chosen in the config, where (Semantic Info) < (Visual Info).

    # Load the model weights.
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)
    # Perform inference.
    # The last parameter, num_query, is 1 because there is only 1 suspect per test set image.
    do_inference(cfg, model, inference_loader, 1)
