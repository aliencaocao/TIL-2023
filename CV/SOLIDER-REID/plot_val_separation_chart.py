import argparse
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
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
    cfg.EXECUTION_MODE = 'plot_val'
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
    added_distances = intra_class_distances + inter_class_distances 

    # cache the distances for later use
    with open(os.path.join(output_dir, 'val_set_distances.pkl'), 'wb') as f:
        pickle.dump((inter_class_distances, intra_class_distances), f)

    # build the histogram for both inter_class_distances and intra_class_distances
    print('Building histogram for inter_class_distances and intra_class_distances')
    max_value_of_both = max(max(inter_class_distances), max(intra_class_distances))
    min_value_of_both = min(min(inter_class_distances), min(intra_class_distances))
    bin_size = 1e-7
    bins = np.arange(min_value_of_both, max_value_of_both + bin_size, bin_size)
    intra_hist, _ = np.histogram(intra_class_distances, bins=bins)
    inter_hist, _ = np.histogram(inter_class_distances, bins=bins)
    
    # calculate the optimal val set threshold that gives highest 'accuracy'
    tp = 0
    tn = np.sum(inter_hist)
    total = np.sum(intra_hist) + tn
    accs = np.array([]) 
    for i, bin in enumerate(bins[:-1]):
        num_intra = intra_hist[i]
        num_inter = inter_hist[i]
        tp += num_intra
        tn -= num_inter
        acc = (tp + tn) / total
        accs = np.append(accs, acc)
    # find the index with the highest accuracy
    max_acc_index = np.argmax(accs)
    optimal_thresh = bins[max_acc_index] + bin_size

    # find the intersection area of the intra and inter histograms
    # find the sum of intra_hist from max_acc_index to the end
    fn = np.sum(intra_hist[max_acc_index+1:])
    # find the sum of inter_hist from the beginning to max_acc_index
    fp = np.sum(inter_hist[:max_acc_index])
    intersection_area = fn + fp

    # find the minimum stationary point of the kde plot of added_distances
    # this is because the test set will see the two distributions (intra and inter) as one distribution (added)
    kde = gaussian_kde(added_distances)
    x = np.linspace(np.min(added_distances), np.max(added_distances), num=2000)
    kde_plot = kde(x)
    gradient = np.gradient(kde_plot)
    # Find the indices where the gradient changes sign (stationary points)
    stationary_indices = np.where(np.diff(np.sign(gradient)))[0]
    # Find the minimum stationary point
    x_minpt = x[stationary_indices][1]

    # calculate the difference between the optimal val set threshold and the minimum stationary point
    delta = optimal_thresh - x_minpt

    print('Plotting the histograms')
    # Plot the histogram for added_distances (All) with green color
    sns.histplot(added_distances, color='green', label='Added Distances', binrange=(min_value_of_both, max_value_of_both))
    # Plot the histogram for inter_class_distances (Dissimilar) with orange color
    sns.histplot(inter_class_distances, color='orange', label='Inter Class Distances', binrange=(min_value_of_both, max_value_of_both))
    # Plot the histogram for intra_class_distances (Similar) with blue color
    sns.histplot(intra_class_distances, color='blue', label='Intra Class Distances', binrange=(min_value_of_both, max_value_of_both))

    # Set labels and title
    plt.xlabel('Distances')
    plt.ylabel('Count')

    # draw a vertical line at the optimal val set threshold
    plt.axvline(x=optimal_thresh, color='black', linestyle='--', label='Intersection Point')
    plt.text(optimal_thresh, 0, f'{optimal_thresh:.15e}', color='black', rotation=90, verticalalignment='bottom', horizontalalignment='center')
    plt.text(optimal_thresh, plt.ylim()[1] * 0.9, 'O', color='black', ha='center')

    # draw a vertical line at the minimum stationary point
    plt.axvline(x_minpt, color='red', linestyle='--')
    plt.text(x_minpt, 0, f'{x_minpt:.15e}', rotation=90, verticalalignment='bottom', horizontalalignment='center')
    plt.text(x_minpt, plt.ylim()[1] * 0.9, 'S', color='black', ha='center')

    # annotate the other metrics
    plt.text(plt.xlim()[1], plt.ylim()[1] * 0.6, f'Accuracy@O: {accs[max_acc_index]}', color='black', ha='center')
    plt.text(plt.xlim()[1], plt.ylim()[1] * 0.5, f'Intersection Area@O: {intersection_area}', color='black', ha='center')
    plt.text(plt.xlim()[1], plt.ylim()[1] * 0.4, f'O-S Delta: {delta}', color='black', ha='center')

    # Add a legend
    plt.legend()

    # Add a title
    if cfg.TEST.RE_RANKING:
        distance_type = 'reranking'
    else:
        distance_type = 'euclidean'

    plt.title(f'Val Set Separation Chart [{distance_type} distance]\nModel: {cfg.TEST.WEIGHT}')

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{distance_type}_separation_chart.png'))

    # Save the metrics to a text file
    with open(os.path.join(output_dir, f'{distance_type}_separation_chart.txt'), 'w') as f:
        f.write(f'Val optimal threshold: {optimal_thresh}\n')
        f.write(f'Val optimal threshold acc: {accs[max_acc_index]}\n\n')
        f.write(f'Val added_distance min point: {x_minpt}\n')
        f.write(f'Val added_distance delta: {delta}\n')
        f.write(f'Val Intersection Area: {intersection_area}\n')