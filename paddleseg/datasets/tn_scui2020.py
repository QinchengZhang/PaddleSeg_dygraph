# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-09 12:36:05
LastEditors: TJUZQC
LastEditTime: 2020-12-09 13:15:30
Description: None
'''
import os

from .dataset import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose

URL = "https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip"


@manager.DATASETS.add_component
class TN_SCUI2020(Dataset):
    """
    TN_SCUI2020 dataset is extraced from Grand-Challenge
    (https://tn-scui2020.grand-challenge.org/Home/).

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory. Default: None
        mode (str): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
    """

    def __init__(self, dataset_root=None, transforms=None, mode='train'):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = 2
        self.ignore_index = 255

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be 'train' or 'val', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME)
        elif not os.path.exists(self.dataset_root):
            self.dataset_root = os.path.normpath(self.dataset_root)
            savepath, extraname = self.dataset_root.rsplit(
                sep=os.path.sep, maxsplit=1)
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=savepath,
                extrapath=savepath,
                extraname=extraname)

        if mode == 'train':
            file_path = os.path.join(self.dataset_root, 'train_list.txt')
        else:
            file_path = os.path.join(self.dataset_root, 'val_list.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(' ')
                if len(items) != 2:
                    raise Exception(
                        "File list format incorrect! It should be"
                        " image_name label_name\\n")
                else:
                    image_path = items[0]
                    grt_path = items[1]
                self.file_list.append([image_path, grt_path])
