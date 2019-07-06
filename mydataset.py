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
from torch.autograd import Variable
import torch.nn.functional as F

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

class MyDataset(Dataset):
    """docstring for MyDataset."""
    def __init__(self, imgs_csv,rois_csv, ther_path,root_dir, transform=None,train=True):
        super(MyDataset, self).__init__()
        self.train = train
        self.imgs = pd.read_csv(imgs_csv)
        self.bb = pd.read_csv(rois_csv)
        self.bb.set_index('id', inplace=True)
        self.root_dir = root_dir
        self.transform = transform
        self.ther_path = ther_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgs.iloc[idx,0])
        if self.train:
            gt_name = img_name.replace('images_train', 'annotations_train')
        else:
            gt_name = img_name.replace('images_test', 'annotations_test')

        gt_name = gt_name.replace('jpg', 'txt')


        image = cv2.imread(img_name)

        bbs = self.bb.loc[idx].reset_index().as_matrix()

        bbs = bbs.astype('float')

        tm = cv2.imread(os.path.join(self.ther_path,self.imgs.iloc[idx,0].replace('visible','lwir')),0)

        if self.train:
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
            gt_boxes_padding = getGTboxesPadding(gt_boxes,cfg.TRAIN.MAX_GTS)
            gt_boxes_padding = torch.from_numpy(gt_boxes_padding)
            # print('Size of all ROIS', all_rois.size())

            fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * cfg.TRAIN.ROI_PER_IMAGE))
            fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
            label,rois,gt_rois = sample_rois_tensor(all_rois, gt_boxes_padding, fg_rois_per_image, cfg.TRAIN.ROI_PER_IMAGE, cfg.NUM_CLASSES)

            sample = {'img_info':img_name,
                      'image': image,
                      'label': label,
                      'bb': rois.numpy(),
                      'tm': tm,
                      'gt': gt_boxes_padding.numpy(),
                      'gt_roi': gt_rois
                      }
        else:
            sample = {'img_info':img_name,
                      'image': image,
                      'label': torch.zeros([1,1]),
                      'bb': bbs[:,:-1],
                      'tm': tm,
                      'gt': np.zeros([1,1]),
                      'gt_roi': torch.zeros([1,1]),
                      }

        if self.transform:
            sample = self.transform(sample)

        return sample

def getSampleDataset(id = None,bz=1,p=0.5,trans=True,train=True):
    '''
        input: id: the index of image in dataset (optional)
               bz: batch size
        output: (dict) sample {'image','bb','tm','img_info','gt'}

        '''
    if train:
        full_transform=transforms.Compose([RandomHorizontalFlip(p),
                                           ToTensor(),
                                           Normalize(cfg.BGR_MEAN,cfg.BGR_STD)])
    else:
        full_transform=transforms.Compose([ToTensor(),
                                           Normalize(cfg.BGR_MEAN,cfg.BGR_STD)])
    if trans is False:
        full_transform = None

    params = {'batch_size': bz,
              'shuffle': cfg.TRAIN.SHUFFLE,
              'num_workers': cfg.TRAIN.NUM_WORKERS}
    print(params)

    if train:
        my_dataset = MyDataset(imgs_csv=cfg.TRAIN.IMGS_CSV,rois_csv=cfg.TRAIN.ROIS_CSV,
        root_dir=cfg.TRAIN.ROOT_DIR, ther_path=cfg.TRAIN.THERMAL_PATH,transform = full_transform,train=train)
    else:
        my_dataset = MyDataset(imgs_csv=cfg.TEST.IMGS_CSV,rois_csv=cfg.TEST.ROIS_CSV,
        root_dir=cfg.TEST.ROOT_DIR, ther_path=cfg.TEST.THERMAL_PATH,transform = full_transform,train=train)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)

    dataiter = iter(dataloader)
    return dataiter

    # if not id:
    #     dataiter = iter(dataloader)
    #     return dataiter.next()
    # else:
    #     return my_dataset[id]


if __name__ == '__main__':
    sample  = getSampleDataset(id = 36999)

    print(sample['image'].shape)
    print(sample['bb'].shape)
    print(sample['label'].shape)
    print(sample['gt'].shape)
    print(sample['gt_roi'].shape)

    print(sample['gt'])
    label = sample['label']
    rois = sample['bb']
    gt_rois = sample['gt_roi']
    # exit()
    #
    # print('='*10)
    # bbox_target_data = compute_targets(rois[:,1:5],gt_rois[:,:4],label)
    #
    label = label.expand(1,128)
    rois = rois.expand(1,128,5)
    gt_rois = gt_rois.expand(1,128,5)
    bbox_target_data = compute_targets_pytorch(rois[:,:, 1:5], gt_rois[:,:,:4])
    print(bbox_target_data)
    print(bbox_target_data.size())
    print(bbox_target_data[0,0])

    exit()
    # bbox_targets, bbox_inside_weights = get_bbox_regression_labels_pytorch(bbox_target_data, label, num_classes=2)
    # #
    # print('=========')
    # print(bbox_targets.size())
    # print(bbox_inside_weights.size())
    # print(bbox_targets)

    bbox_outside_weights = (bbox_inside_weights > 0).float()
    # print(bbox_outside_weights.size())

    label = label.view(-1).long()
    bbox_targets = bbox_targets.view(-1, bbox_targets.size(2))
    bbox_inside_weights = bbox_inside_weights.view(-1, bbox_inside_weights.size(2))
    bbox_outside_weights = bbox_outside_weights.view(-1, bbox_outside_weights.size(2))

    print(label.size())
    print(bbox_targets.size())
    print(bbox_inside_weights.size())
    print(bbox_outside_weights.size())

    # cls_score, bbox_pred = torch.load('out_MSDN_test4.pth')
    # RCNN_loss_cls = F.cross_entropy(cls_score[:2,:], label)
    # print(RCNN_loss_cls.mean())
    # RCNN_loss_bbox = smooth_l1_loss(bbox_pred[:2], bbox_targets, bbox_inside_weights, bbox_outside_weights)
    # print(RCNN_loss_bbox.mean())
