from __future__ import print_function, division
import os, sys, cv2
import torch
import pandas as pd
# from skimage import io
import numpy as np

from torch.utils.data import Dataset, DataLoader
# from my_transforms import *
from torchvision import transforms

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

        bbs = self.bb.loc[idx].iloc[:self.NUM_BBS].reset_index().as_matrix()
        bbs = bbs.astype('float')

        tm = cv2.imread(os.path.join(self.ther_path,self.imgs.iloc[idx,0].replace('visible','lwir')),0)

        gt_boxes = []

        with open(gt_name,'r') as f:
            data = f.readlines()

        for i in range(1,len(data)):
            d = data[i].split()
            #x1,x2,y1,y2,cls
            temp = [int(d[1]),int(d[2]),int(d[1])+int(d[3]),int(d[2])+int(d[4]),1]
            gt_boxes.append(temp)

        gt_boxes_padding = np.zeros((self.MAX_GTS, 5),dtype=np.float)
        if gt_boxes:
            num_gts = len(gt_boxes)
            gt_boxes = np.asarray(gt_boxes,dtype=np.float)

            # gt_boxes_padding = torch.FloatTensor(self.MAX_GTS, 5).zero_()
            gt_boxes_padding[:num_gts,:] = gt_boxes[:num_gts]

        sample = {'img_info':img_name, 'image': image, 'bb': bbs, 'tm':tm, 'gt':gt_boxes_padding}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_train.csv'
    ROIS_CSV = 'mydata/rois_trainKaist_thr70_1.csv'
    full_transform=transforms.Compose([RandomHorizontalFlip(),
                                       ToTensor(),
                                       my_normalize()])
                                  # Normalize(rgb_mean,rgb_std)])
    device = torch.device("cuda:0")
    params = {'batch_size':1,
              'shuffle':True,
              'num_workers':24}


    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)
    dataiter = iter(dataloader)
    sample = dataiter.next()
    # sample = my_dataset[789]
    NUM_BBS = my_dataset.NUM_BBS
    print('NUM_BBS',NUM_BBS)

    # print(sample['image'])
    # print('===================')
    # print(sample['tm'].size())
    # print(torch.max(sample['tm']))

    testDataset(sample,True)
    # img = sample['image']
    # bbb = sample['bb']
    # bbb = bbb.view(-1,5)
    # idx = -1
    # for i,v in enumerate(bbb[:,0]):
    #     if not i%NUM_BBS:
    #         idx = idx + 1
    #     bbb[i,0] = idx
    #
    # img,bbb = img.to(device), bbb.to(device)
    #
    # roi_pool = ROIPool((50, 50), 1)
    # x = roi_pool(img, bbb)
    # print(x.shape)
    #
    # bbb = bbb.cpu().detach().numpy()
    # img = convertTensor2Img(img)
    # img = visualizeRP(img, bbb)
    # cv2.imshow('AAA', img)
    #
    # x = x.type('torch.ByteTensor')
    # x = x.cpu().detach().numpy()
    # for ind,labels in enumerate(x):
    #     print(labels.shape)
    #     cv2.imshow('bbs{}'.format(ind), labels.transpose((1, 2, 0)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]

    # ther = convertTensor2Img(tm,0)
    # labels_output = resizeThermal(tm, bbb)
    # print(labels_output.shape)
        # for ind,labels in enumerate(labels_output):
        #     print(labels.shape)
        #     cv2.imshow('bbs{}'.format(ind), labels)
    # for i,p in enumerate(labels_output):
    #     print(p.shape)
    #     cv2.imshow('bba{}'.format(i), p)
    # tmm = cv2.resize(labels_output, (50,50))
    # cv2.imshow('bbbb', tmm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # exit()
    # bbb=bbb.view(-1, 5)
    # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]
    # #reset id
    # tmp = set()

    # idx = -1
    # print(bbb)
    # # # #
    # criterion = nn.MSELoss()
    #
    # out = roi_pool(tm,bbb)
    # loss = criterion(out, out)
    # print(loss.item())
    # showTensor(out)



    # image = io.imread(img_id)
    # a = torch.from_numpy(image)
    # print(a.view(-1,3).shape)

    # bb = data_frame.iloc[0, 1:-1].as_matrix()
    # bb = bb.astype('float')
    # print(bb)
# print('load batch size')

    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #           sample_batched['bb'])
    #
    #     roi_pool = ROIPool((7, 7), 1.0)
    #     pooled_output = roi_pool(sample_batched['image'])
    #     input()
    #     exit()


# print(len(my_dataset))
# for i in range(len(my_dataset)):
#     print(i,my_dataset[i]['image'].shape)
# sample = my_dataset[0]

# fig = plt.figure()

# for i in range(len(my_dataset)):
#     sample = my_dataset[i]
#
#     print(i, sample['image'].shape, sample['bb'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_img_bb(**sample)
#
#     if i == 3:
#         plt.show()
#         break
# plt.waitforbuttonpress()
