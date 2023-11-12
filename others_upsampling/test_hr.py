# -*- coding: utf-8 -*-
# @Time : 2023/3/4 9:49
# @Author : LiQingCode
# @File : test_hr.py
# @Project : Joint Image Upsampling with Affinity Learning
import copy
import argparse
from torch.autograd import Variable
from module import JUAL
from test_base import *
parser = argparse.ArgumentParser(description='Joint Image Upsampling with Affinity Learning')
parser.add_argument('--task',  type=str, default='GSF',          help='TASK')
parser.add_argument('--model', type=str, default='JUAL', help='model')
args = parser.parse_args()

config = copy.deepcopy(default_config)

config.TASK = args.task
config.SET_NAME = 512

# model
if args.model == 'JUAL':
    model = JUAL(in_channels=9, out_channels=3, is_depth=False)
else:
    print('Not a valid model!')
    exit(-1)

if args.model == 'JUAL':
    model.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, config.TASK, 'net_latest.pth')), strict=False)

config.model = model

def forward(imgs, config):
    x_hr, gt_hr, x_lr, gt_lr = imgs[:4]
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            x_hr, gt_hr, x_lr, gt_lr = x_hr.unsqueeze(0).cuda(), gt_hr.unsqueeze(0).cuda(), x_lr.unsqueeze(0).cuda(), gt_lr.unsqueeze(0).cuda()
    out = config.model(Variable(x_lr), Variable(x_hr), Variable(gt_lr)).data.cpu()
    return out

config.forward = forward

run(config)