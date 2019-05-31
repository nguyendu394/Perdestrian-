from __future__ import print_function, division
import os, sys, cv2
import torch
import pandas as pd
# from skimage import io
import numpy as np

from torch.utils.data import Dataset, DataLoader
from my_transforms import *
from torchvision import transforms
from torchvision.ops import box_iou
from proposal_processing import *
# import torchvision.transforms.functional as TF
# import matplotlib.patches as patches
# from PIL import Image
# import random
# from model.roi_layers import ROIPool

# from image_processing import convertTensor2Img, showBbs,visualizeRP, resizeThermal, flipBoundingBox
# import torch.nn as nn
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
 # mean=(0,485, 0,456, 0,406) and std=(0,229, 0,224, 0,225)

class MyDataset(Dataset):
    """docstring for MyDataset."""
    def __init__(self, imgs_csv,rois_csv, ther_path,root_dir, transform=None):
        super(MyDataset, self).__init__()
        self.imgs = pd.read_csv(imgs_csv)
        self.bb = pd.read_csv(rois_csv)
        self.bb.set_index('id', inplace=True)
        self.root_dir = root_dir
        self.transform = transform
        self.ther_path = ther_path
        self.MAX_GTS = 9
        self.rois_per_image = 128
        self.num_classes = 2
        # print('NUM_BBS:',self.NUM_BBS)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgs.iloc[idx,0])
        gt_name = img_name.replace('images_train', 'annotations_train')
        gt_name = gt_name.replace('jpg', 'txt')

        image = cv2.imread(img_name)

        bbs = self.bb.loc[idx].reset_index().as_matrix()

        bbs = bbs.astype('float')

        tm = cv2.imread(os.path.join(self.ther_path,self.imgs.iloc[idx,0].replace('visible','lwir')),0)

        #create ground-truth
        gt_boxes = []
        with open(gt_name,'r') as f:
            data = f.readlines()
        for i in range(1,len(data)):
            d = data[i].split()
            #x1,x2,y1,y2,cls
            temp = [float(d[1]),float(d[2]),float(d[1])+float(d[3]),float(d[2])+float(d[4]),1]
            gt_boxes.append(temp)

        all_rois = getAllrois(bbs[:,:-1], gt_boxes)
        all_rois = torch.from_numpy(all_rois)
        #padding ground-truth
        gt_boxes_padding = getGTboxesPadding(gt_boxes,self.MAX_GTS)
        gt_boxes_padding = torch.from_numpy(gt_boxes_padding)
        # print('Size of all ROIS', all_rois.size())
        # all_rois,gt_boxes_padding = all_rois.cuda(),gt_boxes_padding.cuda()
        fg_rois_per_image = int(np.round(0.25 * self.rois_per_image))
        label,rois,gt_rois = sample_rois_tensor(all_rois, gt_boxes_padding, fg_rois_per_image, self.rois_per_image, self.num_classes)

        sample = {'img_info':img_name,
                  'image': image,
                  'label': label,
                  'bb': rois,
                  'tm': tm,
                  'gt': gt_boxes_padding.cpu(),
                  'gt_roi': gt_rois
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""
    TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

if __name__ == '__main__':
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_train.csv'
    ROIS_CSV = 'mydata/rois_trainKaist_thr70_MSDN.csv'
    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),
                                       my_normalize()])
                                  # Normalize(rgb_mean,rgb_std)])
    device = torch.device("cuda:0")
    params = {'batch_size':1,
              'shuffle':True,
              'num_workers':24}
    print(params)

    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = None)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)
    dataiter = iter(dataloader)
    # sample = dataiter.next()
    sample = my_dataset[36999]
    num_classes = 2
    rois_per_image = 128
    fg_rois_per_image = int(np.round(0.25 * rois_per_image))

    # print(sample['image'])
    # print(sample['bb'].size())
    all_rois = sample['bb']
    #
    gt_boxes = sample['gt']

    all_rois = torch.from_numpy(all_rois)
    gt_boxes = torch.from_numpy(gt_boxes)
    # all_rois,gt_boxes = all_rois.cuda(),gt_boxes.cuda()
    label,rois,bbox_targets, bbox_inside_weights = sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes)

    print(label[:6])
    print(label.shape)

    print(rois[:6,:])
    print(rois.shape)

    print(bbox_targets[:6,:])
    print(bbox_targets.shape)

    print(bbox_inside_weights[:6,:])
    print(bbox_inside_weights.shape)
