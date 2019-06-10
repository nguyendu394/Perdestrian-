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

def testDataset(sample):
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
    print(gt)

    raw = convertTensor2Img(sam)
    ther = convertTensor2Img(tm,norm=False)
    # print(ther)
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

    # labels_output = labels_output.type('torch.ByteTensor')

    ther = convertTensor2Img(tm,norm=False)
    bbb = bbb.detach().numpy()
    imgg = visualizeRP(ther, bbb)

    cv2.imshow('winname', imgg)
    for ind,labels in enumerate(labels_output):
        p = convertTensor2Img(labels,norm=False)
        cv2.imshow('bbs{}'.format(ind), p)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testRRN_Pretrain(sample,pre):
    device = torch.device("cuda:0")
    RRN_net = MyRRN()
    RRN_net.to(device)
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
    # labels_output = labels_output.to(device)

    sam,bbb,tm = sam.to(device), bbb.to(device), tm.to(device)

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
    device = torch.device("cuda:0")
    sam = sample['image']
    bbs = sample['bb']
    bz = bbs.size(0)
    num = bbs.size(1)

    bbs=bbs.view(-1, 5)
    ind = torch.arange(bz,requires_grad=False).view(-1,1)
    ind = ind.repeat(1,num).view(-1,1)
    bbs[:,0] = ind[:,0]

    sam,bbs = sam.to(device),bbs.to(device)
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
    bbox,score = bbox.cuda(),score.cuda()
    keep = nms(bbox,score,0.5)
    if bbox.size(0) == keep.size(0):
        return True
    else:
        return False

def main():
    pre = 'mymodel/RRN/model27_lr_1e-6_bz_6_NBS_128_norm_epoch_9.pth'
    # sample = getSampleDataset(id = 2027,train=True,bz=1)
    sample = getSampleDataset(train=True,bz=1)

    # print(sample['bb'])
    # testDataset(sample)
    # testROIpool(sample)
    # testResizeThermal(sample)
    testRRN_Pretrain(sample, pre)
if __name__ == '__main__':
    main()
    # cls_dets = torch.Tensor([[],[],[],[],[]]).permute(1,0)
    # print(cls_dets)
    # print(cls_dets.numel())
    #
    #
    # print(a)
    # print(a[1,:,0::4].clamp_(0, im_shape[i, 1]-1))
    # a = torch.randn(2,3,4)
    # print(a)
    # print(a.reshape(-1))
    # b = a.view(1,6,5)
    # print(b)
    # x = a.numpy()
    # b = torch.randn(3)
    # y = b.numpy()
    # print(x.size)
