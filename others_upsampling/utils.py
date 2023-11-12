# -*- coding: utf-8 -*-
# @Time : 2023/3/2 10:08
# @Author : LiQingCode
# @File : utils.py
# @Project : Joint Image Upsampling with Affinity Learning
import numpy as np
import warnings
warnings.simplefilter('ignore')

class Config(object):
    def __init__(self, **params):
        for k, v in params.items():
            self.__dict__[k] = v

def tensor_to_img(tensor, transpose=False):
    im = np.asarray(np.clip(np.squeeze(tensor.numpy()) * 255, 0, 255), dtype=np.uint8)
    if transpose:
        im = im.transpose((1, 2, 0))

    return im