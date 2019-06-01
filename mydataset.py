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

        #run on CUDA
        # all_rois,gt_boxes_padding = all_rois.cuda(),gt_boxes_padding.cuda()
        fg_rois_per_image = int(np.round(0.25 * self.rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        label,rois,gt_rois = sample_rois_tensor(all_rois, gt_boxes_padding, fg_rois_per_image, self.rois_per_image, self.num_classes)

        sample = {'img_info':img_name,
                  'image': image,
                  'label': label,
                  'bb': rois.numpy(),
                  'tm': tm,
                  'gt': gt_boxes_padding.numpy(),
                  'gt_roi': gt_rois
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

def getDataLoader(id = None,bz=1,p=0.5,trans=True):
    '''
        input: id: the index of image in dataset (optional)
               bz: batch size
        output: (dict) sample {'image','bb','tm','img_info','gt'}

        '''
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_train.csv'
    ROIS_CSV = 'mydata/rois_trainKaist_thr70_MSDN.csv'
    full_transform=transforms.Compose([RandomHorizontalFlip(p),
                                       ToTensor(),
                                       my_normalize()])
                                  # Normalize(rgb_mean,rgb_std)])
    if trans is False:
        full_transform = None

    device = torch.device("cuda:0")
    params = {'batch_size':bz,
              'shuffle':True,
              'num_workers':24}
    print(params)

    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)
    if not id:
        return dataloader
    else:
        return my_dataset[id]



if __name__ == '__main__':
    dataloader = getDataLoader(bz=2)

    dataiter = iter(dataloader)
    sample = dataiter.next()


    print(sample['image'].shape)
    print(sample['bb'].shape)
    print(sample['label'].shape)
    print(sample['gt'].shape)
    print(sample['gt_roi'].shape)

    print(sample['gt'])
    label = sample['label']
    rois = sample['bb']
    gt_rois = sample['gt_roi']
    #
    # print('='*10)
    # bbox_target_data = compute_targets(rois[:,1:5],gt_rois[:,:4],label)
    #
    # label = label.expand(1,128)
    # rois = rois.expand(1,128,5)
    # gt_rois = gt_rois.expand(1,128,5)
    bbox_target_data = compute_targets_pytorch(rois[:,:, 1:5], gt_rois[:,:,:4])
    # print(bbox_target_data[0][1])
    bbox_targets, bbox_inside_weights = get_bbox_regression_labels_pytorch(bbox_target_data, label, num_classes=2)
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

    cls_score, bbox_pred = torch.load('out_MSDN_test4.pth')
    RCNN_loss_cls = F.cross_entropy(cls_score[:2,:], label)
    print(RCNN_loss_cls.mean())
    RCNN_loss_bbox = smooth_l1_loss(bbox_pred[:2], bbox_targets, bbox_inside_weights, bbox_outside_weights)
    print(RCNN_loss_bbox.mean())
