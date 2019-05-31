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
        self.gt = None
        self.NUM_BBS = 4
        self.MAX_GTS = 9
        print('NUM_BBS:',self.NUM_BBS)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgs.iloc[idx,0])
        gt_name = img_name.replace('images_train', 'annotations_train')
        gt_name = gt_name.replace('jpg', 'txt')

        image = cv2.imread(img_name)

        bbs = self.bb.loc[idx].iloc[:].reset_index().as_matrix()

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
        #padding ground-truth
        gt_boxes_padding = getGTboxesPadding(gt_boxes,self.MAX_GTS)

        sample = {'img_info':img_name, 'image': image, 'bb': all_rois, 'tm':tm, 'gt':gt_boxes_padding}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
    all_rois,gt_boxes = all_rois.cuda(),gt_boxes.cuda()
    label,rois = sample_rois_tensor(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes)
    print(label)
    print(label.shape)
    print(rois[:6,:])
    print(rois.shape)
