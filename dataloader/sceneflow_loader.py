from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os

from dataloader.utils import read_text_lines
from dataloader.file_io import read_disp,read_img
# from utils.utils import read_text_lines
# from utils.file_io import read_disp,read_img
from skimage import io, transform
import numpy as np

# DOWN: modify the dataset to vkitti's two file list
class StereoDataset(Dataset):
    def __init__(self, 
                 train_gt_dir,
                 train_rgb_dir,
                 train_depthgt_list,
                 train_rgb_list,
                #  test_depthgt_list,
                #  test_rgb_list,
                 dataset_name='vkitti',
                 mode='train',
                 save_filename=False,
                #  load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.train_gt_dir = train_gt_dir
        self.train_rgb_dir = train_rgb_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_depthgt_list = train_depthgt_list
        self.train_rgb_list = train_rgb_list
        # self.test_depthgt_list = test_depthgt_list
        # self.test_rgb_list = test_rgb_list
        # self.img_size=(1242, 375)
        # self.scale_size =(576,960)
        self.img_size=(512, 384)
        

        vkitti_finalpass_dict = {
            'train_gt':  self.train_depthgt_list,
            'train_rgb': self.train_rgb_list,
            # 'test_gt': self.test_depthgt_list,
            # 'test_rgb': self.test_rgb_list 
        }

        dataset_name_dict = {
            'vkitti': vkitti_finalpass_dict,
        }

        self.samples = []

        key_gt = self.mode + '_gt'
        key_rgb  = self.mode + '_rgb'
        gt_filenames = dataset_name_dict[dataset_name][key_gt]
        rgb_filenames = dataset_name_dict[dataset_name][key_rgb]

        lines_gt = read_text_lines(gt_filenames)
        lines_rgb = read_text_lines(rgb_filenames)

        for line_gt, line_rgb in zip(lines_gt, lines_rgb):
            
            # splits = line.split()

            # left_img, right_img = splits[:2]
            # gt_disp = None if len(splits) == 2 else splits[2]
            assert line_rgb == line_gt, "unmatched data pairs"

            sample = dict()

            # if self.save_filename:
            #     sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(train_rgb_dir, line_rgb)
            sample['disp'] = os.path.join(train_gt_dir, line_gt)

            sample['pseudo_disp'] = None

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
        # sample['img_right'] = read_img(sample_path['right'])
    
    
        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        sample['gt_disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]

        if self.mode=='test':
            # img_left = transform.resize(sample['img_left'], [576,960], preserve_range=True)
            # img_right = transform.resize(sample['img_right'], [576,960], preserve_range=True)
            img_left = sample['img_left']
            # img_right = sample['img_right']
            
            img_left = img_left.astype(np.float32)
            # img_right = img_right.astype(np.float32)
            
            sample['img_left'] = img_left
            # sample['img_right'] = img_right

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    # def get_scale_size(self):
    #     return self.scale_size