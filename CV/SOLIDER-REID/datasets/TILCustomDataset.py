# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class TILCustomDataset(BaseImageDataset):

    # This is a modified version of the Market1501 dataset structure.
    # Note that when in training, all the personID and cameraIDs are 1-indexed. This must be converted into a 0-indexed format.
    # However, when in inference, note that the identities of the images are not known. Thus, personID is not applicable.
    # Also note that we create one TILCustomDataset for each test set image. Thus, cameraID is not applicable.

    dataset_dir = ''

    def __init__(self, root='', EXECUTION_MODE=False, **kwargs):
        super(TILCustomDataset, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.EXECUTION_MODE = EXECUTION_MODE

        if EXECUTION_MODE == 'training':
            print('Loading TILCustomDataset in training mode.')
            self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        elif EXECUTION_MODE == 'inference':
            print('Loading TILCustomDataset in testing mode.')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        if EXECUTION_MODE == 'training':
            # If we are in training, process the train directory.
            # This will relabel the personID and cameraID from 1-indexed to 0-indexed.
            train = self._process_dir(self.train_dir, relabel=True)
        else:
            # If we are in testing mode, then there is no train set.
            # Set the train set to be empty.
            train = []
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        # Check if all files are available before going deeper.

        # Check if dataset directory exists.
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

        # Check if the train directory exists, only if we are training.
        if self.EXECUTION_MODE == 'training':
            if not osp.exists(self.train_dir):
                raise RuntimeError("'{}' is not available".format(self.train_dir))

        # Check if the query directory exists. The query directory is used regardless of whether EXECUTION_MODE is training or inference.
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))

        # Check if the gallery directory exists. The gallery directory is used regardless of whether EXECUTION_MODE is training or inference.
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        # Find the paths of all images in the directory.
        img_paths = glob.glob(osp.join(dir_path, '*.jpg')) + glob.glob(osp.join(dir_path, '*.png'))
        if self.EXECUTION_MODE == 'training' or self.EXECUTION_MODE == 'plot_val':
            # If we are in training mode, then we need to relabel the personID and cameraID from 1-indexed to 0-indexed.
            pattern = re.compile(r'([-\d]+)_c(\d+)')
            pid_container = set()
            for img_path in sorted(img_paths):
                pid, _ = map(int, pattern.search(img_path).groups())
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            dataset = []
            for img_path in sorted(img_paths):
                pid, camid = map(int, pattern.search(img_path).groups())
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((img_path, pid, camid, 1))
            return dataset
        elif self.EXECUTION_MODE == 'inference':
            if dir_path == self.query_dir:
                # If we are in inference mode and the directory is the query directory
                # then there is only one image, which is the query suspect image.
                # The file path to this image is simply img_paths[0].
                # The personID is irrelevant as the identity of the suspect is unknown.
                # The cameraID is also irrelevant as a TILCustomDataset is instantiated for each test set image.
                # The last number, trackID, is set to an arbitrary value, in this case 1.
                return [(img_paths[0], -1, 0, 1)]
            else:
                # If we are in inference mode and the directory is the gallery directory
                # then there are multiple images, which are the cropped-out bounding boxes from our model preds.
                # The personID is irrelevant as the identity of the suspect is unknown.
                # The cameraID is also irrelevant as a TILCustomDataset is instantiated for each test set image.
                # The last number, trackID, is set to an arbitrary value, in this case 1.
                dataset = []
                for img_path in sorted(img_paths):
                    dataset.append((img_path, -1, 0, 1))
                return dataset