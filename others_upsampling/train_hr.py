# -*- coding: utf-8 -*-
# @Time : 2023/3/2 10:25
# @Author : LiQingCode
# @File : train_hr.py
# @Project : Joint Image Upsampling with Affinity Learning
import copy
import argparse
from train_base import *
from module import JUAL

parser = argparse.ArgumentParser(description='Joint Image Upsampling with Affinity Learning')
parser.add_argument('--task',  type=str, default='GSF',          help='TASK')
parser.add_argument('--model', type=str, default='JUAL', help='model')
parser.add_argument('--resume', action='store_true', default=False, help='keep going last epoch')
args = parser.parse_args()

config = copy.deepcopy(default_config)

config.TASK = args.task
config.N_EPOCH = 20
config.DATA_SET = 512
config.N_START = 0

# model
if args.model == 'JUAL':
    config.model = JUAL(in_channels=9, out_channels=3, is_depth=False)
else:
    print('Not a valid model!')
    exit(-1)

if args.model == 'JUAL':
    if args.resume:
        state_dict = torch.load(
            os.path.join('../checkpoints',
                         config.TASK,
                         'net_latest.pth')
        )
        config.model.load_state_dict(state_dict, strict=False)

def forward(imgs, config):
    x_hr, gt_hr, x_lr, gt_lr = imgs[:4]
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            x_hr, gt_hr, x_lr, gt_lr = x_hr.cuda(), gt_hr.cuda(), x_lr.cuda(), gt_lr.cuda()
    return config.model(Variable(x_lr), Variable(x_hr), Variable(gt_lr)), gt_hr

config.forward = forward
config.exceed_limit = lambda size: size[0]*size[1] > 2048**2
config.clip = 0.01

run(config)