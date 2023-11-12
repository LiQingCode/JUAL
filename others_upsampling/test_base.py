# -*- coding: utf-8 -*-
# @Time : 2023/3/4 9:50
# @Author : LiQingCode
# @File : test_base.py
# @Project : Joint Image Upsampling with Affinity Learning
import os
import torch
from tqdm import tqdm
from dataset import SuDataset
from utils import tensor_to_img, Config
from skimage.io import imsave

default_config = Config(
    TASK = None,
    SET_NAME = 512,
    #################### CONSTANT #####################
    IMG = '../dataset/MIT-FiveK',
    LIST = '../train_test_list',
    MODEL_PATH = '../checkpoints',
    RESULT_PATH = '../results',
    BATCH = 1,
    LOW_SIZE = 64,
    GPU = 0,
    # model
    model = None,
    # forward
    forward = None,
    # save paths
    save_paths = None,
    # compare paths
    compare_paths = None
)

def run(config):
    assert config.TASK is not None, 'Please set task name: TASK'

    assert config.save_paths is None     and config.compare_paths is None or \
           config.save_paths is not None and config.compare_paths is not None

    if config.save_paths is None:
        config.save_paths = [os.path.join(config.RESULT_PATH, config.TASK)]
    else:
        config.save_paths = [os.path.join(config.RESULT_PATH, config.TASK, path)
                                                                    for path in config.save_paths]

    for path in config.save_paths:
        if not os.path.isdir(path):
            os.makedirs(path)
    # data set
    test_data = SuDataset(config.IMG,
                          os.path.join(
                              config.LIST,
                              config.TASK,
                              'test_{}.csv'.format(config.SET_NAME)
                          ),
                          low_size=config.LOW_SIZE)

    # GPU
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            config.model.cuda()

    # test
    i_bar = tqdm(total=len(test_data), desc='#Images')
    for idx, imgs in enumerate(test_data):
        name = os.path.basename(test_data.get_path(idx))
        imgs_new = config.forward(imgs, config)
        for path, img_new in zip(config.save_paths, imgs_new):
            imsave(os.path.join(path, name), tensor_to_img(img_new, transpose=True))

        i_bar.update()