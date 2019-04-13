import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from skimage import io, color

def equalizeHist(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    claheImg = clahe.apply(img)
    # claheImg = cv2.equalizeHist(img)
    claheImg = cv2.fastNlMeansDenoising(claheImg,None,4,7,21)

    return claheImg
def visualizeRP(img,bbs, fm = 'ltrb',c = 255):
    '''
    Visualize region proposal
    input: img (numpy) WxHxC
           bbs (tensor) ltwh/ltrb bzxnx5
    '''
    bbs = bbs.type('torch.IntTensor')
    bbs = bbs.cpu()
    bbs = bbs.view(-1,5)
    bbs = bbs.detach().numpy()

    for d in bbs:
        id,l,t,r,b = d
        # print(l,t,r,b)
        if fm == 'ltrb':
            img = cv2.rectangle(img,(l,t),(r,b),(0,c,0),1)

        elif fm == 'ltwh':
            img = cv2.rectangle(img,(l,t),(r+l,b+t),(0,c,0),1)
            # img = cv2.rectangle(img,(l,t),(r+l,b+t),(0,c,255),2)
    return img

def createImgsFilesName(path,name_file):
    '''
        Create a txt file with name of images in dir path
    '''
    a = list(os.walk(path))
    a[0][2].sort()
    print('Lenght: ',len(a[0][2]))
    for name in a[0][2]:
        with open(name_file,'a') as f:
            f.write(name+'\n')
    print('Done!')

def convertRoisACF2CSV(path,new):
    '''
        Convert the file of ACF into CSV files
        input: path:  of txt file
                new: the new file

    '''
    with open(path,'r') as f:
        data = f.readlines()

    for i in data:
        id,l,t,w,h,s = i.split(',')
        id = int(id) - 1
        r = round(float(l) + float(w),3)
        b = round(float(t) + float(h),3)
        with open(new,'a') as f:
            f.write('{},{},{},{},{}\n'.format(id,l,t,r,b))

    print('Done!')

def convertTensor2Img(out):
    '''
    Visualize a Tensors
    input: a Tensors on cpu (1x1xhxw)
    output: a image(numpy)
    '''
    out = out.type('torch.ByteTensor')
    out = out.cpu()
    out = out.detach().numpy()
    if len(out.shape) == 4:
        img1 = out[0].transpose((1, 2, 0))
    elif len(out.shape) == 3:
        img1 = out.transpose((1, 2, 0))
    # img2 = out[-1].transpose((1, 2, 0))
    # img3 = out[3].transpose((1, 2, 0))
    #
    return img1

def showBbs(img, bbs):
    '''
        Draw region proposal into images
        input: image(numpy)
               bbs(Tensor) nxlxtxrxb
    '''
    for d in bbs:
        l,t,r,b = d
        img = cv2.rectangle(img,(int(l),int(t)),(int(r),int(b)),(0,255,0),1)

    cv2.imshow('sss', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img = cv2.imread('I01793.jpg')
    bbs_csv = 'mydata/rois_train_thr70.csv'
    bbs = pd.read_csv(bbs_csv)
    bbs.set_index('id', inplace=True)
    bbs = bbs.loc[1793].as_matrix()
    showBbs(img, bbs)



if __name__ == '__main__':
    # main()
    path = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/train/images_train_tm'
    img = io.imread(os.path.join(path,'set05_V000_lwir_I02899.jpg'))
    # img = img[:,:,::-1]
    # img = color.rgb2gray(img)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    # img = img_as_ubyte(img)
    # img = img[:,:,::-1]
    img = equalizeHist(img)
    io.imshow(img)
    plt.show()
    # cv2.imshow('aa',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
