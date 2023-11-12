import numpy as np
import glob
from torch.utils.data import Dataset
from PIL import Image

class dataset_load(Dataset):
    """NYUv2、Lu、Middlebury、Sintel"""
    def __init__(self, root_dir, scale=8, type_down='nearest', train=True, transform=None, name='NYUv2'):
        self.name = name
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.type_down = type_down
        if train:
            self.depths = np.load('%s/train_depth_split.npy' % root_dir)
            self.images = np.load('%s/train_images_split.npy' % root_dir)
        else:
            self.depths = []
            self.images = []
            for gt_name in sorted(glob.glob('{}/{}/gt/*.npy'.format(root_dir, name))):
                gt_img = np.load(gt_name)
                rgb_img = np.load(gt_name.replace('gt', 'rgb'))
                self.depths.append(gt_img)
                self.images.append(rgb_img)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]

        h, w = depth.shape
        s = self.scale
        if self.type_down == 'nearest':
            target = np.array(
                Image.fromarray(depth).resize((w // s, h // s), Image.NEAREST).resize((w, h), Image.BICUBIC))
        else:
            target = np.array(
                Image.fromarray(depth).resize((w // s, h // s), Image.BICUBIC).resize((w, h), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth, 2)).float()
            target = self.transform(np.expand_dims(target, 2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth}

        return sample

class NYU_v2_datset(Dataset):
    """NYUDataset."""
    def __init__(self, root_dir, scale=8, type_down='nearest', train=True, transform=None, name='NYUv2'):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.name = name
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.type_down = type_down
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        
        h, w = depth.shape
        s = self.scale
        if self.type_down == 'nearest':
            target = np.array(
                Image.fromarray(depth).resize((w // s, h // s), Image.NEAREST).resize((w, h), Image.BICUBIC))
        else:
            target = np.array(
                Image.fromarray(depth).resize((w // s, h // s), Image.BICUBIC).resize((w, h), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth, 2)).float()
            target = self.transform(np.expand_dims(target, 2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth}
        
        return sample