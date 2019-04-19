from __future__ import print_function, division
import os, sys, cv2
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
# import matplotlib.patches as patches
# from PIL import Image
import random
from model.roi_layers import ROIPool
from image_processing import equalizeHist, convertTensor2Img, showBbs,visualizeRP, resizeThermal
import torch.nn as nn
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
NUM_BBS = 4
rgb_mean = (0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225)

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

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgs.iloc[idx,0])
        image = io.imread(img_name)

        bbs = self.bb.loc[idx].iloc[:NUM_BBS].reset_index().as_matrix()
        bbs = bbs.astype('float')

        tm = io.imread(os.path.join(self.ther_path,self.imgs.iloc[idx,0].replace('visible','lwir')))
        sample = {'image': image, 'bb': bbs, 'tm':tm}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.
    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
              ``short``).
    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """
    def __init__(self, mode='BGR'):
        self.mode = mode

    def __call__(self, sample):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL Image: Image converted to PIL Image.
        """
        sample['image'] = TF.to_pil_image(sample['image'], self.mode)
        sample['tm'] = TF.to_pil_image(sample['tm'], self.mode)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbs, tm = sample['image'], sample['bb'], sample['tm']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # bbs = bbs.transpose((1,0))

        while bbs.shape[0] < NUM_BBS:
            bbs = np.concatenate((bbs,bbs))
        image = np.array(image)
        image = image.transpose((2, 0, 1))

        tm = np.array(tm,dtype='uint8')
        tm = equalizeHist(tm)
        # print(type(tm[0][0][0]))
        # print(type(tm))
        tm = tm.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).type('torch.FloatTensor'),
                'bb': torch.from_numpy(bbs[:NUM_BBS]).type('torch.FloatTensor'),
                'tm': torch.from_numpy(tm).type('torch.FloatTensor')}

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        img = TF.to_pil_image(sample['image'], 'RGB')
        ther = TF.to_pil_image(sample['tm'], 'RGB')
        if random.random() < self.p:
            sample['image'] = TF.hflip(img)
            sample['tm'] = TF.hflip(ther)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        sample['image'] = TF.normalize(sample['image'], self.mean, self.std, self.inplace)
        sample['tm'] = TF.normalize(sample['tm'], self.mean, self.std, self.inplace)

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__ == '__main__':
    THERMAL_PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm/'
    ROOT_DIR = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train'
    IMGS_CSV = 'mydata/imgs_train.csv'
    ROIS_CSV = 'mydata/rois_train_thr70.csv'
    my_transform = ToTensor()
    full_transform=transforms.Compose([RandomHorizontalFlip(p=5),
                                       ToTensor(),])
                                  # Normalize(rgb_mean,rgb_std)])

    device = torch.device("cuda:0")
    params = {'batch_size':2,
              'shuffle':True,
              'num_workers':24}


    my_dataset = MyDataset(imgs_csv=IMGS_CSV,rois_csv=ROIS_CSV,
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)
    dataiter = iter(dataloader)
    sample = dataiter.next()
    # sample = my_dataset[789]

 # Lặp qua bộ dữ liệu huấn luyện nhiều lần
        # for i, data in enumerate(dataloader):
        #     count = count + 1
        #     print(count)



    # sample = my_dataset[37900]
    #
    # # data_frame = pd.read_csv('data/1256.csv')
    # # img_id = data_frame.iloc[0,0]
    #
    # dataiter = iter(dataloader)
    # sample = dataiter.next()
    sam = sample['image']
    bbb = sample['bb']
    tm = sample['tm']


    # raw = convertTensor2Img(sam,1)
    # ther = convertTensor2Img(tm,0)
    # bbs = visualizeRP(raw, bbb)

    #
    # cv2.imshow('raw',raw)
    # cv2.imshow('thermal', ther)
    # cv2.imshow('bbs', bbs)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # tm = sample['tm']
    # print(sam.shape)
    # print(bbb.shape)
    # print(tm.shape)
    bbb = bbb.view(-1,5)
    idx = -1
    for i,v in enumerate(bbb[:,0]):
        if not i%NUM_BBS:
            idx = idx + 1
        bbb[i,0] = idx
    # bbb[:, 0] = bbb[:, 0] - bbb[0, 0]

    # ther = convertTensor2Img(tm,0)
    labels_output = resizeThermal(tm, bbb)
    print(labels_output.shape)
    # for ind,labels in enumerate(labels_output):
    #     print(labels.shape)
    #     cv2.imshow('bbs{}'.format(ind), labels)
    # for i,p in enumerate(labels_output):
    #     print(p.shape)
        # cv2.imshow('bba{}'.format(i), p)
    # # tmm = cv2.resize(labels_output, (50,50))
    # cv2.imshow('bbbb', tmm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    roi_pool = ROIPool((50, 50), 1)
    tm, bbb = tm.to(device),bbb.to(device)
    labels_output = roi_pool(tm,bbb)
    print(labels_output.shape)
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
