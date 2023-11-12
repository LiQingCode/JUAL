import argparse

import torch
from torch import nn
from nyu_dataloader import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from module import JUAL
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
import utility
import os

parser = argparse.ArgumentParser()
parser.add_argument('--guide_channels', type=int, default=3, help='guide channels')
parser.add_argument('--scale', type=int, default=8, help='scale factor')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--model',  default='JUAL', help='choose model JUAL')
parser.add_argument('--lr',  default='0.001', type=float, help='learning rate')
parser.add_argument('--result',  default='../checkpoints/depth_map', help='learning rate')
parser.add_argument('--epoch',  default=20, type=int, help='max epoch')
downsampling_method = 'nearest'
opt = parser.parse_args()
print(opt)

result_root = '%s/%s'%(opt.result, opt.model)
if not os.path.exists(result_root): os.mkdir(result_root)

logging.basicConfig(filename='%s/train-%s-%s.log'%(result_root, downsampling_method, opt.scale), format='%(asctime)s %(message)s', level=logging.INFO)

model_cspn = JUAL(in_channels=7, out_channels=1, is_depth=True, mode=downsampling_method, scale=opt.scale)
net = torch.nn.DataParallel(model_cspn, device_ids=[0]).cuda()

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.2)
net.train()

data_transform = transforms.Compose([
    transforms.ToTensor()
])
ToPIL = transforms.ToPILImage()
nyu_dataset = NYU_v2_datset(root_dir='../dataset', scale=opt.scale, type_down=downsampling_method,
                            transform=data_transform)
dataloader = torch.utils.data.DataLoader(nyu_dataset, batch_size=opt.batch_size, shuffle=True)

def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    a = a*(minmax[1]-minmax[0]) + minmax[1]
    b = b*(minmax[1]-minmax[0]) + minmax[1]
    
    return np.sqrt(np.mean(np.power(a-b, 2)))

def test_pick(net, test_name='NYUv2'):
    device = torch.device('cuda')
    net.eval()
    sum_rmse = []
    idx = 1
    for gt_name in tqdm(sorted(glob.glob('../dataset/{}/gt/*.npy'.format(test_name)))):
        gt_img = np.load(gt_name)
        rgb_img = np.load(gt_name.replace('gt', 'rgb'))
        tmp_gt = utility.mod_crop(gt_img, modulo=opt.scale)
        if downsampling_method == 'nearest':
            tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
            lr_img = utility.get_lowers(tmp_gt, factor=opt.scale, mode='nearest')
        else:
            tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
            lr_img = utility.get_lowers(tmp_gt, factor=opt.scale, mode='bicubic')

        lr_up = utility.get_lowers(lr_img, factor=1 / opt.scale, mode='bicubic')
        lr_img, gt_img = np.expand_dims(lr_img, 0), np.expand_dims(gt_img, 0)

        lr_up = np.expand_dims(lr_up, 0)
        if opt.guide_channels == 1:
            rgb_img = np.expand_dims(utility.rgb2gray(rgb_img), 2)

        rgb_img = np.float32(np.transpose(rgb_img, axes=(2, 0, 1))) / 255.

        gt_img, rgb_img = utility.mod_crop(gt_img, modulo=opt.scale), utility.mod_crop(rgb_img, modulo=opt.scale)

        lr_img, lr_up, gt_img, rgb_img = utility.np_to_tensor(lr_img, lr_up, gt_img, rgb_img)

        lr_img, lr_up, gt_img, rgb_img = lr_img.unsqueeze(0), lr_up.unsqueeze(0), gt_img.unsqueeze(
            0), rgb_img.unsqueeze(0)
        lr_img, lr_up, gt_img, rgb_img = lr_img.to(device), lr_up.to(device), gt_img.to(device), rgb_img.to(device)

        out = net(x_lr=Variable(lr_up.contiguous()), x_hr=Variable(rgb_img.contiguous()))

        if out.shape[1] == 3:
            gt_img = torch.cat([gt_img, gt_img, gt_img], 1)

        idx = idx + 1
        if test_name == 'NYUv2':
            mul_ratio = 100
        elif test_name == 'Sintel':
            mul_ratio = 255
        else:
            mul_ratio = 1

        rmse, _ = utility.root_mean_sqrt_error(im_pred=out.contiguous(), im_true=gt_img.contiguous(), border=6,
                                               mul_ratio=mul_ratio, is_train=False)
        sum_rmse.append(rmse)
    return sum_rmse

min_rmse = 1
max_epoch = opt.epoch
for epoch in range(max_epoch):
    net.train()
    running_loss = 0.0
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        optimizer.zero_grad()
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
        out = net(Variable(target), Variable(guidance))
        loss = criterion(out, gt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.data.item()
        if idx % 50 == 0:
            running_loss /= 50
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch+1, running_loss))
            t.refresh()

    logging.info('epoch:%d' % (epoch + 1))
    test_set = ['MiddleBury', 'Lu', 'NYUv2', 'Sintel']
    for test_name in test_set:
        rmse = test_pick(net, test_name=test_name)
        mean_rmse = np.mean(rmse)
        logging.info('%s:%f'%(test_name, mean_rmse))
        print('%s:%f'%(test_name, mean_rmse))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            torch.save(net.state_dict(), "%s/Ours_%s_%sx_best_%s"%(result_root, downsampling_method, opt.scale, test_name))
    torch.save(net.state_dict(), "%s/epoch_%s_Ours_%s_%sx" % (result_root, epoch+1, downsampling_method, opt.scale))