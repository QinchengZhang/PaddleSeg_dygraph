# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math

import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar
from PIL import Image


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map

def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.

    """
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]
    

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    mkdir(added_saved_dir)
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    mkdir(pred_saved_dir)
    pred_one_channel_saved_dir = os.path.join(save_dir, 'one_channel_pseudo_color_prediction')
    mkdir(pred_one_channel_saved_dir)

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            im = cv2.imread(im_path)
            ori_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            if aug_pred:
                pred = infer.aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = os.path.basename(im_path)
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/':
                im_file = im_file[1:]

            # save added image
            added_image = utils.visualize.visualize(im_path, pred, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred)
            pred_saved_path = os.path.join(pred_saved_dir,
                                           im_file.rsplit(".")[0] + ".png")
            pred_mask.save(pred_saved_path)

            # save one channel pseudo color prediction
            pred_im_one_channel = Image.fromarray(pred)
            pred_im_one_channel = pred_im_one_channel.convert('P')
            colormap = get_color_map_list(256)
            pred_im_one_channel.putpalette(colormap)
            pred_one_channel_saved_path = os.path.join(pred_one_channel_saved_dir, im_file)
            pred_im_one_channel.save(pred_one_channel_saved_path.replace('jpg', 'png'))


            progbar_pred.update(i + 1)
