import cv2
import torch
from torchvision import transforms
from my_transforms import *
from torch.utils.data import Dataset, DataLoader
from mydataset import MyDataset
from image_processing import convertTensor2Img, visualizeRP,resizeThermal
from RRN import MyRRN
import torchvision.ops.roi_pool as ROIPool
from torchvision.ops import nms

def getDataLoader(bz=1,p=0.5,trans=True):
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
    return dataloader

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

    # for ind,labels in enumerate(labels_output):
    #     print(labels.shape)
    #     cv2.imshow('bbs{}'.format(ind), labels)
    raw = convertTensor2Img(sam)
    ther = convertTensor2Img(tm)
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
    out = labels_output.cpu()
    out = out.detach().numpy()
    ther = convertTensor2Img(tm)
    bbb = bbb.detach().numpy()
    imgg = visualizeRP(ther, bbb)


    cv2.imshow('winname', imgg)
    for ind,labels in enumerate(out):
        p = labels.transpose((1, 2, 0))

        if p.dtype == np.float32:
            p = (p*255).astype(np.uint8)
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

    # print(labels_output.size())
    # labels_output = labels_output.type('torch.ByteTensor')
    out = labels_output.cpu()
    out = out.detach().numpy()
    # plt.imshow(imgg)
    # plt.show()
    sam,bbb,tm = sam.to(device), bbb.to(device), tm.to(device)

    out_RRN = RRN_net(sam,bbb)
    out_RRN = out_RRN.cpu().detach().numpy()

    bbb = bbb.cpu().detach().numpy()

    ther = convertTensor2Img(tm)
    imgg = visualizeRP(ther, bbb)
    # print(imgg.dtype)
    cv2.imshow('aa',imgg)

    for ind,labels in enumerate(out_RRN):
        # print('output')
        p = labels.transpose((1, 2, 0))
        if p.dtype == np.float32:
            p = (p*255).astype(np.uint8)
        cv2.imshow('rrn{}'.format(ind), p)

    for ind,labels in enumerate(out):
        p = labels.transpose((1, 2, 0))
        if p.dtype == np.float32:
            p = (p*255).astype(np.uint8)
        cv2.imshow('bbs{}'.format(ind), p)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  testROIpool(sample):
    device = torch.device("cuda:0")
    sam = sample['image']
    bbs = sample['bb']
    bz = bbb.size(0)
    num = bbb.size(1)
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


    simg = simg.cpu().detach().numpy()
    for ind,labels in enumerate(simg):
        p = labels.transpose((1, 2, 0))
        if p.dtype == 'float32':
            p = (p*255).astype(np.uint8)

        cv2.imshow('rrn{}'.format(ind), p)

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
    pre = 'models/model24/model24_lr_1e-6_bz_6_NBS_128_norm_epoch_9.pth'

    dataloader = getDataLoader(p=5)
    dataiter = iter(dataloader)
    sample = dataiter.next()
    testDataset(sample)
    # testROIpool(sample)
    # testResizeThermal(sample)
    # testRRN_Pretrain(sample, pre)
if __name__ == '__main__':
    main()
    # a = torch.randn(5)
    # x = a.numpy()
    # b = torch.randn(3)
    # y = b.numpy()
    # print(x.size)
