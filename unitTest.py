import cv2
import torch
from torchvision import transforms
from my_transforms import *
from torch.utils.data import Dataset, DataLoader
from mydataset import MyDataset
from image_processing import convertTensor2Img, visualizeRP,resizeThermal
from RRN import MyRRN
import torchvision.ops.roi_pool as ROIPool


def getSampleDataset(id = None):
    '''
        input: id: the index of image in dataset (optional)
        output: (dict) sample {'image','bb','tm','img_info','gt'}
                (scaler) maximum number of bbox in a image})
        '''
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
    root_dir=ROOT_DIR, ther_path=THERMAL_PATH,transform = full_transform)
    print(my_dataset.__len__())
    dataloader = DataLoader(my_dataset, **params)
    dataiter = iter(dataloader)
    if id:
        sample = my_dataset[id]
    else:
        sample = dataiter.next()

    return sample, my_dataset.NUM_BBS

def testDataset(sample,norm = True):
    print(sample['img_info'])

    sam = sample['image']
    tm = sample['tm']
    bbs = sample['bb'][:,:,:-1]
    gt = sample['gt']
    # print(bbs.size())
    # print(gt.size())

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



def testResizeThermal(sample,NUM_BBS,bz):
    sam = sample['image']
    bbb = sample['bb'][:,:,:-1]
    tm = sample['tm']
    gt = sample['gt']

    bbb = bbb.cpu()
    bbb = bbb.view(-1,5)


    gt = gt.cpu()
    gt = gt.view(-1,5)
    gt = gt.detach().numpy()


    ind = torch.arange(bz).view(-1,1)
    ind = ind.repeat(1,NUM_BBS).view(-1,1)
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

def testRRN_Pretrain(sample,pre,num, norm=True,bz=1):
    device = torch.device("cuda:0")
    RRN_net = MyRRN()
    RRN_net.to(device)
    RRN_net.load_state_dict(torch.load(pre))
    NUM_BBS = num
    sam = sample['image']
    bbb = sample['bb']
    tm = sample['tm']
    gt = sample['gt']
    bbb=bbb.view(-1, 5)

    ind = torch.arange(bz,requires_grad=False).view(-1,1)
    ind = ind.repeat(1,NUM_BBS).view(-1,1)
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

    ther = convertTensor2Img(tm,norm)
    imgg = visualizeRP(ther, bbb)
    # print(imgg.dtype)
    cv2.imshow('aa',imgg)

    for ind,labels in enumerate(out_RRN):
        # print('output')
        p = labels.transpose((1, 2, 0))*255
        cv2.imshow('rrn{}'.format(ind), p.astype(np.uint8))

    for ind,labels in enumerate(out):
        p = labels.transpose((1, 2, 0))*255
        cv2.imshow('bbs{}'.format(ind), p.astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def  testROIpool(sample,NUM_BBS,bz=1):
    device = torch.device("cuda:0")
    sam = sample['image']
    bbs = sample['bb'][:,:,:-1] #get rid of score

    bbs=bbs.view(-1, 5)
    ind = torch.arange(bz,requires_grad=False).view(-1,1)
    ind = ind.repeat(1,NUM_BBS).view(-1,1)
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

def testNMS():
    sample,num = getSampleDataset()
    print(sample['bb'])
def main():
    pre = 'models/model23/model23_lr_1e-9_bz_6_NBS_128_norm_epoch_3.ptx'
    sample,num = getSampleDataset()
    # testDataset(sample)
    # testROIpool(sample,num)
    testResizeThermal(sample, num, 1)

if __name__ == '__main__':
    main()
