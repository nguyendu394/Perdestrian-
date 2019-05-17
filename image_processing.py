import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from skimage import io, color, exposure,restoration
from PIL import Image
import torch

def equalizeHist(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    # img = clahe.apply(img)
    # img = color.rgb2hsv(img)
    gray = color.rgb2gray(img)
    gray = exposure.equalize_adapthist(gray,clip_limit=0.03)
    gray = restoration.denoise_tv_chambolle(gray , weight=0.1)

    # claheImg = cv2.equalizeHist(img)
    # img = cv2.fastNlMeansDenoising(img,None,4,7,21)

    return gray

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

    for ind,i in enumerate(data):
        id,l,t,w,h,s = i.split(',')
        id = int(id) - 1
        r = round(float(l) + float(w),3)
        b = round(float(t) + float(h),3)
        with open(new,'a') as f:
            l = max(float(l),0)
            t = max(float(t),0)
            f.write('{},{},{},{},{}\n'.format(id,l,t,r,b))

    print('Done!')

def convertTensor2Img(out,k):
    '''
    Visualize a Tensors
    input: a Tensors on cpu (1x1xhxw)
    output: a image(numpy)
    '''
    if k:
        out = out.type('torch.ByteTensor')
    out = out.cpu()
    out = out.detach().numpy()
    # print(type(out[0][0][0][0]))
    if len(out.shape) == 4:
        img = out[0].transpose((1, 2, 0))
    elif len(out.shape) == 3:
        img = out.transpose((1, 2, 0))
    return img

def resizeThermal(img,rois):
    tm_croppeds = []

    rois = rois.type('torch.IntTensor')
    img = img.cpu()
    # img = img.type('torch.ByteTensor')
    img = img.detach().numpy()

    for roi in rois:
        id,y1,x1,y2,x2 = roi
        tm_cropped = img[id].transpose((1, 2, 0))
        # print(tm_cropped.shape)
        tm_cropped = tm_cropped[x1:x2,y1:y2]
        tm_cropped = cv2.resize(tm_cropped, (50,50))
        # exit()
        tm_croppeds.append(np.expand_dims(tm_cropped,axis=0))
    return torch.from_numpy(np.array(tm_croppeds)).type('torch.FloatTensor')

def showBbs(img, bbs):
    '''
        Draw region proposal into images
        input: image(numpy)
               bbs(Tensor) nx4 (l,t,r,b)
    '''
    for d in bbs:
        l,t,r,b = d
        img = cv2.rectangle(img,(int(l),int(t)),(int(r),int(b)),(0,255,0),1)

    return img

def readLogFile(path):
    print(path)
    val = []
    st = 10
    with open(path,'r') as f:
        data = f.readlines()
    iters = list(range(10, (len(data)+1)*10,10))

    for d in data:
        val.append(float(d.split()[3]))
    return iters, val

def visualizeErrorLoss(true_txt, false_txt=None):
    if false_txt:
        f_iter, f_val = readLogFile(false_txt)
        plt.plot(f_iter, f_val, color='g')
    t_iter, t_val = readLogFile(true_txt)

    plt.plot(t_iter[500:], t_val[500:], color='orange')
    plt.xlabel('iter')
    plt.ylabel('L2 loss');
    plt.title('error loss')
    plt.show()

def main():
    # img = cv2.imread('I01793.jpg')
    bbs_txt = '../ngoc/toolbox/detector/models/Dets_TestK_All_with_Kaist_Thr70.txt'
    bbs_csv = 'mydata/rois_testKaist_thr70_0.csv'

    convertRoisACF2CSV(bbs_txt, bbs_csv)
if __name__ == '__main__':
    # createImgsFilesName('/storageStudents/K2015/duyld/dungnm/dataset/KAIST/test/images_test', 'mydata/imgs_test.txt')
    # main()
    # print('./models/model14/log14.txt')
    # true_txt = './models/model15/log15.txt'
    test_txt = './test2_model21_epoch7.txt'
    visualizeErrorLoss(test_txt)
