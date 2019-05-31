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
        self.NUM_BBS = 128
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

        all_rois = self.getAllrois(bbs[:,:-1], gt_boxes)
        #padding ground-truth
        # gt_boxes_padding = self.getGTboxesPadding(gt_boxes)

        sample = {'img_info':img_name, 'image': image, 'bb': all_rois, 'tm':tm, 'gt':np.asarray(gt_boxes)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getGTboxesPadding(self,gt_boxes):
        gt_boxes_padding = np.zeros((self.MAX_GTS, 5),dtype=np.float)
        if gt_boxes:
            num_gts = len(gt_boxes)
            gt_boxes = np.asarray(gt_boxes,dtype=np.float)

            # gt_boxes_padding = torch.FloatTensor(self.MAX_GTS, 5).zero_()
            gt_boxes_padding[:num_gts,:] = gt_boxes[:num_gts]
        return gt_boxes_padding

    def getAllrois(self,all_rois,gt_boxes):
        ''' get all bbox include predicted bbox, gt_boxes and jitter gt bboxes
            input: all_rois(numpy)(N,4)[l,t,r,b]
                   gt_boxes(numpy)(K,5)[l,t,r,b,cls]
            output: numpy (M,4) M = N+K*2
        '''

        if gt_boxes:
            # print('START')
            # print(all_rois)
            # print('='*10)
            # print(gt_boxes)
            # print('='*10)
            # print(jit_gt_boxes)
            # print('='*10)
            gt_boxes = np.asarray(gt_boxes)
            jit_gt_boxes = self._jitter_gt_boxes(gt_boxes)
            zeros = np.zeros((gt_boxes.shape[0] * 2, 1), dtype=gt_boxes.dtype)
            all_rois = np.vstack((all_rois, np.hstack((zeros,np.vstack((gt_boxes[:, :-1], jit_gt_boxes[:, :-1]))))))
            # print(all_rois)
            # print('END')
            return all_rois
        else:
            return all_rois

    def _jitter_gt_boxes(self,gt_boxes, jitter=0.05):
        """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        """
        jittered_boxes = gt_boxes.copy()
        ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
        hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
        width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
        height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
        jittered_boxes[:, 0] += width_offset
        jittered_boxes[:, 2] += width_offset
        jittered_boxes[:, 1] += height_offset
        jittered_boxes[:, 3] += height_offset

        return jittered_boxes


def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)

    overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

    max_overlaps, gt_assignment = torch.max(overlaps, 2)

    batch_size = overlaps.size(0)
    num_proposal = overlaps.size(1)
    num_boxes_per_img = overlaps.size(2)

    offset = torch.arange(0, batch_size)*gt_boxes.size(1)
    offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

    # changed indexing way for pytorch 1.0
    labels = gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)

    labels_batch = labels.new(batch_size, rois_per_image).zero_()
    rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
    gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
    # Guard against the case when an image has fewer than max_fg_rois_per_image
    # foreground RoIs
    for i in range(batch_size):

        fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
        fg_num_rois = fg_inds.numel()

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
        bg_num_rois = bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
            # See https://github.com/pytorch/pytorch/issues/1868 for more details.
            # use numpy instead.
            #rand_num = torch.randperm(fg_num_rois).long().cuda()
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

            # Seems torch.rand has a bug, it will generate very large number and make an error.
            # We use numpy rand instead.
            #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
            rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
            bg_inds = bg_inds[rand_num]

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
            rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = rois_per_image
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
            rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

            bg_inds = bg_inds[rand_num]
            bg_rois_per_this_image = rois_per_image
            fg_rois_per_this_image = 0
        else:
            raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

        # The indices that we're selecting (both fg and bg)
        keep_inds = torch.cat([fg_inds, bg_inds], 0)

        # Select sampled values from various arrays:
        labels_batch[i].copy_(labels[i][keep_inds])

        # Clamp labels for the background RoIs to 0
        if fg_rois_per_this_image < rois_per_image:
            labels_batch[i][fg_rois_per_this_image:] = 0

        rois_batch[i] = all_rois[i][keep_inds]
        rois_batch[i,:,0] = i

        gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

    bbox_target_data = self._compute_targets_pytorch(
            rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])

    bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

    return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
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
    sample = my_dataset[4572]


    # print(sample['image'])
    # print(sample['bb'].size())
    # print(sample['bb'])
    print(sample['bb'].shape)
    #
    print(sample['gt'])
