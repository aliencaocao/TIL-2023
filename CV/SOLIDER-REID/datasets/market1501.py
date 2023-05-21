# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = ''

    def __init__(self, root='', verbose=True, pid_begin = 0, TEST_MODE=False, **kwargs):
        super(Market1501, self).__init__()
        self.TEST_MODE = TEST_MODE
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        if TEST_MODE:
            self.test_dir = osp.join(self.dataset_dir, 'test')
            self.test_query_dir = osp.join(self.dataset_dir, 'suspects')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        if TEST_MODE:
            test = self._process_dir(self.test_dir, relabel=False)
            test_query = self._process_dir(self.test_query_dir, relabel=False)

        if verbose:
            print("=> TIL CUSTOM Market1501 loaded")
            if not self.TEST_MODE:
                self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        if TEST_MODE:
            self.test = test
            self.test_query = test_query

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        # self.num_test_pids, self.num_test_imgs, self.num_test_cams, self.num_test_vids = self.get_imagedata_info(self.test)
        # self.num_test_query_pids, self.num_test_query_imgs, self.num_test_query_cams, self.num_test_query_vids = self.get_imagedata_info(self.test_query)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg')) + glob.glob(osp.join(dir_path, '*.png'))
        TEST_PATH = dir_path == self.test_dir or dir_path == self.test_query_dir
        if not TEST_PATH:
            pattern = re.compile(r'([-\d]+)_c(\d)')
            pid_container = set()
            for img_path in sorted(img_paths):
                pid, _ = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            if not TEST_PATH:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                # assert 0 <= pid <= 1501  # pid == 0 means background
                # assert 1 <= camid <= 6
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
            else:
                dataset.append(img_path)
        return dataset
