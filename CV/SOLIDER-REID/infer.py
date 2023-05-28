import warnings
warnings.filterwarnings(lineno=20, action='ignore', category=UserWarning)

import argparse
import os
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference, do_batch_inference
from utils.logger import setup_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReID Inference')
    parser.add_argument('--config_file', default="", help='Path to YAML config file.', type=str)
    parser.add_argument('--batched', default=False, help='batched or not', type=bool)
    parser.add_argument('opts',
                        help='Modify config options using the command line.',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.EXECUTION_MODE = 'inference' if not args.batched else 'batch_inference'

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger('reid', output_dir, if_train=False)
    logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(
        cfg,
        num_class=2,
        camera_num=1,
        view_num=1,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT
    )

    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)

    if cfg.EXECUTION_MODE == 'inference':
        results = []
        for test_set_img_folder in tqdm(os.listdir(cfg.DATASETS.ROOT_DIR)):
            subcfg = deepcopy(cfg)
            subcfg.DATASETS.ROOT_DIR = os.path.join(cfg.DATASETS.ROOT_DIR, test_set_img_folder)

            inference_loader = make_dataloader(subcfg)
            # NOTE: ALL THREE PARAMS, num_class, camera_num, view_num, ARE UNUSED.
            # YOU CAN PUT 999 FOR ALL OF THEM AND THE MODEL WILL STILL PRODUCE THE SAME RESULTS.
            # num_class is only used in training to determine the size of the classifier head.
            # camera_num and view_num are ignored in the build_transformer function!,
            # Nevertheless, I have put sensible values for all of them.

            # Semantic weight is the ratio of (Semantic Info) / (Visual Info).
            # Since ReID is a visual task, we want to weight the visual info more.
            # Thus, a semantic weight is 0.2 is chosen in the config, where (Semantic Info) < (Visual Info).
             # Load the model weights.

            # The last parameter, num_query, is 1 because there is only 1 suspect per test set image.
            curr_results = do_inference(cfg, model, inference_loader, 1, cfg.TEST.THRESHOLD)
            results.extend(curr_results)
    else:  # batched_inference
        inference_loader = make_dataloader(cfg)
        # num_query equal to number of suspects which is 1699
        results = do_batch_inference(cfg, model, inference_loader, len(os.listdir(os.path.join(cfg.DATASETS.ROOT_DIR, 'query'))), cfg.TEST.THRESHOLD)
    results_df = pd.DataFrame(results, columns=['file_path', 'result'])
    results_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'results.csv'), index=False)