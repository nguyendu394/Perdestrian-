import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def equalizeHist(img):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    claheImg = clahe.apply(img)
    # claheImg = cv2.equalizeHist(img)
    claheImg = cv2.fastNlMeansDenoising(claheImg,None,4,7,21)

    return claheImg

def visualizeRP(img,bbs, fm = 'ltrb',c = 255):
    '''
    Visualize region proposal
    input: img (numpy) WxHxC
           bbs (numpy) ltwh/ltrb
    '''
    for d in bbs:
        id,l,t,r,b = d
        if format == 'ltrb':
            img = cv2.rectangle(img,(l,t),(r,b),(0,c,0),1)
        elif format == 'ltwh':
            img = cv2.rectangle(img,(l,t),(r+l,b+t),(0,c,0),1)
            # img = cv2.rectangle(img,(l,t),(r+l,b+t),(0,c,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img,name,(l,t-5),font, 0.5,(0,255,0),1,cv2.LINE_AA)
    return img

def createImgsFilesName(path,name_file):
    '''
        Create a txt file with name of images in dir path
    '''
    a = list(os.walk(ROOT_DIR))
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

def showTensor(out):
    '''
    Visualize a Tensors
    input: a Tensors on cpu (1x1xhxw)
    '''
    out = out.type('torch.ByteTensor')
    out = out.cpu()
    out = out.detach().numpy()
    img1 = out[0].transpose((1, 2, 0))
    # img2 = out[-1].transpose((1, 2, 0))
    # img3 = out[3].transpose((1, 2, 0))
    #
    #
    cv2.imshow('sss1', img1)
    # cv2.imshow('sss2', img2)
    # cv2.imshow('sss3', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # PATH = '/storageStudents/K2015/duyld/dungnm/dataset/KAIST/images/set00/V000/visible/'
    convertRoisACF2CSV('mydata/Dets_TH2_Thr-70_Cal0005.txt','mydata/rois_set00_v000.txt')
