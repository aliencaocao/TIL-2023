import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import get_distance_distributions
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
    cfg.EXECUTION_MODE = 'plotting'
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

    val_loader, num_query = make_dataloader(cfg)
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

    inter_class_distances, intra_class_distances = get_distance_distributions(cfg, model, val_loader, num_query)
    # Plot the histogram for inter_class_distances (Dissimilar) with orange color
    sns.histplot(inter_class_distances, color='orange', label='Inter Class Distances')
    # Plot the histogram for intra_class_distances (Similar) with blue color
    sns.histplot(intra_class_distances, color='blue', label='Intra Class Distances')

    # # Calculate the intersection point of the two KDEs
    # kde1 = stats.gaussian_kde(inter_class_distances)
    # kde2 = stats.gaussian_kde(intra_class_distances)
    # x = np.linspace(min(min(inter_class_distances), min(intra_class_distances)), max(max(inter_class_distances), max(intra_class_distances)), 1000)
    # intersection_point = x[np.argmax(kde1(x) - kde2(x) < 0)]

    # Set labels and title
    plt.xlabel('Distances')
    plt.ylabel('Count')

    # # Draw a vertical line at the intersection point
    # plt.axvline(x=intersection_point, color='black', linestyle='--', label='Intersection Point')
    # plt.text(intersection_point, plt.ylim()[1] * 0.9, f'Intersection: {intersection_point:.2f}', color='black', ha='center')

    # Add a legend
    plt.legend()

    # Add a title
    if cfg.TEST.RE_RANKING:
        distance_type = 'reranking'
    else:
        distance_type = 'euclidean'

    plt.title(f'Separation Chart [{distance_type} Distance]\nModel: {cfg.TEST.WEIGHT}')

    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{distance_type}_separation_chart.png'))
