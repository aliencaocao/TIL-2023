import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pickle
from tqdm import tqdm
from copy import deepcopy
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

matplotlib.use('Agg')
sns.set_theme(style='whitegrid')

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
    cfg.EXECUTION_MODE = 'inference'

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

    model = make_model(
        cfg,
        num_class=1, # Unused, can put any arbitrary value
        camera_num=1, # Unused, can put any arbitrary value
        view_num=1, # Unused, can put any arbitrary value
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT
    )
    # Semantic weight is the ratio of (Semantic Info) / (Visual Info).
    # Since ReID is a visual task, we want to weight the visual info more.
    # Thus, a semantic weight is 0.2 is chosen in the config, where (Semantic Info) < (Visual Info).

    # Load the model weights.
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)

    collated_distances = []
    for test_set_img_folder in tqdm(os.listdir(cfg.DATASETS.ROOT_DIR)):
        subcfg = deepcopy(cfg)
        subcfg.DATASETS.ROOT_DIR = os.path.join(cfg.DATASETS.ROOT_DIR, test_set_img_folder)

        inference_loader = make_dataloader(subcfg)
        # no threshold as we want the raw distance matrix
        curr_dist_mat = do_inference(cfg, model, inference_loader, 1, None, output_dist_mat=True)
        collated_distances.extend(curr_dist_mat)

    with open(os.path.join(output_dir, 'collated_test_set_distances.pkl'), 'wb') as f:
        pickle.dump(collated_distances, f)

    max_distance = max(collated_distances)
    min_distance = min(collated_distances)

    if cfg.TEST.RE_RANKING:
        distance_type = 'reranking'
    else:
        distance_type = 'euclidean'

    # Plot the histogram of distances
    sns.histplot(collated_distances, binrange=(min_distance, max_distance))

    plt.title(f'Test Set Separation Chart [{distance_type} Distance]\nModel: {cfg.TEST.WEIGHT}')

    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{distance_type}_test_set_separation_chart.png'))
