import cv2
import torch
from torchvision import transforms
from my_transforms import *
from torch.utils.data import Dataset, DataLoader
from mydataset import MyDataset, getSampleDataset
from image_processing import convertTensor2Img, visualizeRP,resizeThermal
from RRN import MyRRN
import torchvision.ops.roi_pool as ROIPool
from torchvision.ops import nms

def testDataset(sample,drawgt=None):
    print(sample['img_info'])

    sam = sample['image']
    tm = sample['tm']
    bbs = sample['bb']
    gt = sample['gt']
    print('sam',sam.size())
    print('tm',tm.size())
    print('bbs',bbs.size())
    print('gt',gt.size())

    bbs = bbs.cpu()
    bbs = bbs.view(-1,5)
    bbs = bbs.detach().numpy()

    gt = gt.cpu()
    gt = gt.view(-1,5)
    gt = gt.detach().numpy()

    raw = convertTensor2Img(sam)
    ther = convertTensor2Img(tm,norm=False)
    # print(ther)
    if not drawgt:
        gt = None
    draw_bbs = visualizeRP(raw, bbs,gt)

    # img,bboxes = flipBoundingBox(raw, bbs)
    # draw_flip_bbs = visualizeRP(img, bboxes,gt)
    print('size of raw',raw.shape)
    print('size of ther',ther.shape)
    print('size of bbs',draw_bbs.get().shape)
    cv2.imshow('raw',raw)
    cv2.imshow('thermal', ther)
    cv2.imshow('bbs', draw_bbs)
    # cv2.imshow('flip_bbs', draw_flip_bbs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testResizeThermal(sample):
    sam = sample['image']
    bbb = sample['bb']
    tm = sample['tm']
    print(sample['img_info'])
    gt = sample['gt']

    bbb = bbb.cpu()
    bz = bbb.size(0)
    num = bbb.size(1)
    bbb = bbb.view(-1,5)


    gt = gt.cpu()
    gt = gt.view(-1,5)
    gt = gt.detach().numpy()


    ind = torch.arange(bz).view(-1,1)
    ind = ind.repeat(1,num).view(-1,1)
    bbb[:,0] = ind[:,0]

    labels_output = resizeThermal(tm, bbb)
    print(labels_output.shape)
    # labels_output = labels_output.type('torch.ByteTensor')
    exit()
    ther = convertTensor2Img(tm,norm=False)
    bbb = bbb.detach().numpy()
    imgg = visualizeRP(ther, bbb[:9])

    cv2.imshow('winname', imgg)
    for ind,labels in enumerate(labels_output[:9]):
        p = convertTensor2Img(labels,norm=False)
        cv2.imshow('bbs{}'.format(ind), p)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testRRN_Pretrain(sample,pre):
    RRN_net = MyRRN()
    RRN_net.to(cfg.DEVICE)
    RRN_net.load_state_dict(torch.load(pre))

    sam = sample['image']
    bbb = sample['bb']
    tm = sample['tm']
    gt = sample['gt']
    bz = bbb.size(0)
    num = bbb.size(1)
    bbb = bbb.view(-1, 5)

    ind = torch.arange(bz,requires_grad=False).view(-1,1)
    ind = ind.repeat(1,num).view(-1,1)
    bbb[:,0] = ind[:,0]

    labels_output = resizeThermal(tm, bbb)
    # labels_output = labels_output.to(cfg.DEVICE)

    sam,bbb,tm = sam.to(cfg.DEVICE), bbb.to(cfg.DEVICE), tm.to(cfg.DEVICE)

    out_RRN = RRN_net(sam,bbb)
    # out_RRN = out_RRN.cpu().detach().numpy()

    bbb = bbb.cpu().detach().numpy()

    ther = convertTensor2Img(tm,norm=False)
    imgg = visualizeRP(ther, bbb)
    # print(imgg.dtype)
    cv2.imshow('aa',imgg)

    for ind,labels in enumerate(out_RRN[:4]):
        # print('output')
        p = convertTensor2Img(labels,norm=False)
        cv2.imshow('rrn{}'.format(ind), p)

    for ind,labels in enumerate(labels_output[:4]):
        p = convertTensor2Img(labels,norm=False)
        cv2.imshow('bbs{}'.format(ind), p)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  testROIpool(sample):
    sam = sample['image']
    bbs = sample['bb']
    bz = bbs.size(0)
    num = bbs.size(1)

    bbs=bbs.view(-1, 5)
    ind = torch.arange(bz,requires_grad=False).view(-1,1)
    ind = ind.repeat(1,num).view(-1,1)
    bbs[:,0] = ind[:,0]

    sam,bbs = sam.to(cfg.DEVICE),bbs.to(cfg.DEVICE)
    simg = ROIPool(sam,bbs,(50,50),1)


    ther = convertTensor2Img(sam)

    bbs = bbs.cpu().detach().numpy()
    img = visualizeRP(ther, bbs)
    cv2.imshow('raw',img)


    # simg = simg.cpu().detach().numpy()

    for ind,labels in enumerate(simg):
        rois = convertTensor2Img(labels)
        cv2.imshow('rrn{}'.format(ind), rois)

    cv2.waitKey()
    cv2.destroyAllWindows()

def testNMS(bbs):
    # idx = bbs[:,:,0].view(-1)
    bbox = bbs[:,:,1:-1].view(-1,4)
    score = bbs[:,:,-1].view(-1)
    bbox,score = bbox.to(cfg.DEVICE),score.to(cfg.DEVICE)
    keep = nms(bbox,score,0.5)
    if bbox.size(0) == keep.size(0):
        return True
    else:
        return False

def main():
    pre = 'mymodel/RRN/model27_lr_1e-6_bz_6_NBS_128_norm_epoch_9.pth'
    # sample = getSampleDataset(train=False,bz=1)
    dataiter = getSampleDataset(train=True,bz=1)
    for sample in dataiter:
        name = sample['img_info']
        img = sample['image']

        gt = sample['gt']
        gt = gt.cpu()
        gt = gt.view(-1,5)
        gt = gt.detach().numpy()

        img = convertTensor2Img(img)
        img = visualizeRP(img, bbs=None,gt=gt)

        cv2.imshow('A',img)
        cv2.waitKey()
        # input('Enter to next image')
    cv2.destroyAllWindows()

    # print(sample['img_info'])
    # print(sample['bb'].size())
    # testDataset(sample,drawgt=True)
    # testROIpool(sample)
    # testResizeThermal(sample)
    # testRRN_Pretrain(sample, pre)
if __name__ == '__main__':
    main()
    #34975 ['/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test/set10_V000_visible_I02300.jpg']
