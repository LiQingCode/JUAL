# -*- coding: utf-8 -*-
import glob
import math
import tqdm
import numpy as np
import torch
import os
import utility
from option import args
from module import JUAL
from torchvision import transforms
args.scale = 8
args.down_type = 'nearest'
method = 'JUAL'
device = torch.device('cpu' if args.cpu else 'cuda')

model = torch.nn.DataParallel(JUAL(in_channels=7, out_channels=1, is_depth=True, mode=args.down_type, scale=args.scale), device_ids=[0]).cuda()
device_id = torch.cuda.current_device()
ToPIL = transforms.ToPILImage()
load_name = '../checkpoints/depth_map/{}/Ours_{}_{}x'.format(method, args.down_type, args.scale)
checkpoint = torch.load(load_name)
model.load_state_dict(checkpoint, strict=False)
model.eval()

test_set = ['Middlebury', 'NYUv2', 'Lu', 'Sintel']
for test_name in test_set:
    sum_rmse = []
    idx = 1
    time_run = []
    result_root = '../results/depth_map/%s/%s/%s-%s' % (test_name, method, args.scale, args.down_type)
    if not os.path.exists(result_root): os.makedirs(result_root)
    for gt_name in tqdm.tqdm(sorted(glob.glob('../dataset/{}/gt/*.npy'.format(test_name)))):
        gt_img = np.load(gt_name)
        rgb_img = np.load(gt_name.replace('gt', 'rgb'))
        module = max(int(math.pow(2, 1 + args.num_pyramid)), args.scale)
        tmp_gt = utility.mod_crop(gt_img, modulo=module)
        if args.down_type == 'nearest':
            tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
            lr_img = utility.get_lowers(tmp_gt, factor=args.scale, mode=args.down_direction)
        else:
            tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
            lr_img = utility.get_lowers(tmp_gt, factor=args.scale, mode='bicubic')
        lr_up = utility.get_lowers(lr_img, factor=1 / args.scale, mode='bicubic')
        lr_img, gt_img = np.expand_dims(lr_img, 0), np.expand_dims(gt_img, 0)
        lr_up = np.random.normal(0, 5/255, lr_up.shape) + lr_up
        lr_up = np.clip(lr_up, 0, 1)
        lr_up = np.expand_dims(lr_up, 0)
        if args.guide_channels == 1:
            rgb_img = np.expand_dims(utility.rgb2gray(rgb_img), 2)

        rgb_img = np.float32(np.transpose(rgb_img, axes=(2, 0, 1))) / 255.

        gt_img, rgb_img = utility.mod_crop(gt_img, modulo=module), utility.mod_crop(rgb_img, modulo=module)

        lr_img, lr_up, gt_img, rgb_img = utility.np_to_tensor(lr_img, lr_up, gt_img, rgb_img)

        lr_img, lr_up, gt_img, rgb_img = lr_img.unsqueeze(0), lr_up.unsqueeze(0), gt_img.unsqueeze(0), rgb_img.unsqueeze(0)
        lr_img, lr_up, gt_img, rgb_img = lr_img.to(device), lr_up.to(device), gt_img.to(device), rgb_img.to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = model(x_lr=lr_up.contiguous(), x_hr=rgb_img.contiguous())
        end.record()
        torch.cuda.synchronize()
        time_run_single = start.elapsed_time(end)
        time_run.append(time_run_single)
        if out.shape[1] == 3:
            gt_img = torch.cat([gt_img, gt_img, gt_img], 1)
        save_out = torch.squeeze(out, 0)
        ToPIL(save_out).save('%s/%d.png' % (result_root, idx))
        idx = idx + 1
        if test_name == 'NYUv2':
            mul_ratio = 100
        elif test_name == 'Sintel':
            mul_ratio = 255
        else:
            mul_ratio = 1

        rmse, _ = utility.root_mean_sqrt_error(im_pred=out.contiguous(), im_true=gt_img.contiguous(), border=6, mul_ratio=mul_ratio, is_train=False)
        sum_rmse.append(rmse)

    print('{}: {:.2f}: {:.6f}'.format(test_name, np.mean(sum_rmse), np.mean(time_run)))
