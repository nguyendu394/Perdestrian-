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
               bbs(Tensor) nx4 (l,t,r,b)
    '''
    for d in bbs:
        l,t,r,b = d
        img = cv2.rectangle(img,(int(l),int(t)),(int(r),int(b)),(0,255,0),1)

    return img


def main():
    img = cv2.imread('I01793.jpg')
    bbs_csv = 'mydata/rois_train_thr70.csv'
    bbs = pd.read_csv(bbs_csv)
    bbs.set_index('id', inplace=True)
    bbs = bbs.loc[1793].as_matrix()
    showBbs(img, bbs)

def readLogFile(text):
    val = []
    st = 10
    with open(text,'r') as f:
        data = f.readlines()
    iters = list(range(10, (len(data)+1)*10,10))
    for d in data:
        val.append(float(d.split()[3]))
    return iters, val


def visualizeErrorLoss(true_txt, false_txt):
    t_iter, t_val = readLogFile(true_txt)
    f_iter, f_val = readLogFile(false_txt)
    print(len(t_iter), len(f_iter))

    plt.plot(t_iter, t_val, color='g')
    plt.plot(f_iter, f_val, color='orange')
    plt.xlabel('iter')
    plt.ylabel('L2 loss');
    plt.title('error loss')
    plt.show()

if __name__ == '__main__':
    # main()
    true_txt = './models/model6/log6.txt'
    false_txt = './models/model7/log7.txt'
    visualizeErrorLoss(true_txt,false_txt)
