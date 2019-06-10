import torchvision.transforms.functional as TF
import numpy as np
import random
from image_processing import equalizeHist,flipBoundingBox
import torch

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
        info = sample['img_info']
        image = sample['image']
        bbs = sample['bb']
        tm = sample['tm']
        gt = sample['gt']
        label = sample['label']
        gt_roi = sample['gt_roi']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # bbs = bbs.transpose((1,0))

        # while bbs.shape[0] < NUM_BBS:
        #     bbs = np.concatenate((bbs,bbs))

        image = np.array(image)
        image = image.transpose((2, 0, 1))

        tm = equalizeHist(tm)
        tm = np.expand_dims(tm,axis=0)
        # tm = tm.transpose((2, 0, 1))

        return {'img_info': info,
                'image': torch.from_numpy(image).type('torch.FloatTensor'),
                'bb': torch.from_numpy(bbs).type('torch.FloatTensor'),
                'tm': torch.from_numpy(tm).type('torch.FloatTensor'),
                'gt': torch.from_numpy(gt).type('torch.FloatTensor'),
                'gt_roi': gt_roi.type('torch.FloatTensor'),
                'label': label.type('torch.FloatTensor')
                }

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
        # img = TF.to_pil_image(sample['image'], 'RGB')
        # ther = TF.to_pil_image(sample['tm'], 'RGB')
        # if random.random() < self.p:
        #     sample['image'] = TF.hflip(img)
        #     sample['tm'] = TF.hflip(ther)
        if random.random() < self.p:
            sample['image'], sample['bb'],  sample['gt'] = flipBoundingBox(sample['image'],sample['bb'], sample['gt'])
            sample['tm'] = sample['tm'][:,::-1]

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
        normimg = sample['image']/255.
        sample['image'] = TF.normalize(normimg, self.mean, self.std, self.inplace)
        # sample['tm'] = TF.normalize(sample['tm'], self.mean, self.std, self.inplace)

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
