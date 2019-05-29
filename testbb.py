import numpy as np
import cv2
from image_processing import showBbs
# from sklearn import preprocessing
import torch

ID = 'I01445'
GT_PATH = './mydata/{}.txt'.format(ID)
#PRED_PATH = './my_mAP/predicted/{}.txt'.format(ID)
IMG_PATH = './mydata/imgs/{}.jpg'.format(ID)
# arr = ['person','people','person?','cyclist']
fm = 'ltrb'
def getCoord(path):
    res = []
    with open(path,'r') as f:
        for line in f:
            print(line)
            eles = line.split(',')
            d = {}
            d['l'] = int(float(eles[1]))
            d['t'] = int(float(eles[2]))
            d['r'] = int(float(eles[3]))
            d['b'] = int(float(eles[4]))
            d['c'] = eles[0]
            res.append(d)
    return res

def drawBB(img, coods, c = 255, format='ltwh'):
    for d in coods:
        l = d['l']
        t = d['t']
        r = d['r']
        b = d['b']
        name = d['c']
        if format == 'ltrb':
            img = cv2.rectangle(img,(l,t),(r,b),(0,c,0),1)
        elif format == 'ltwh':
            img = cv2.rectangle(img,(l,t),(r+l,b+t),(0,c,0),1)
            # img = cv2.rectangle(img,(l,t),(r+l,b+t),(0,c,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img,name,(l,t-5),font, 0.5,(0,255,0),1,cv2.LINE_AA)
    return img

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1['l'],box2['l'])
    yi1 = max(box1['t'],box2['t'])
    xi2 = min(box1['r'],box2['r'])
    yi2 = min(box1['b'],box2['b'])
    # inter_area = np.multiply(yi2-yi1,xi2-xi1)
    inter_area = np.multiply(yi2-yi1+1,xi2-xi1+1)
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = np.multiply(box1['b']-box1['t']+1,box1['r']-box1['l']+1)
    box2_area = np.multiply(box2['b']-box2['t']+1,box2['r']-box2['l']+1)
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###

    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area/union_area
    ### END CODE HERE ###

    return iou

def main():
    # print(IMG_PATH)
    img = cv2.imread(IMG_PATH)
    # cv2.imshow('raw', img)
    print('shape: {}'.format(img.shape))
    gt_coo = getCoord(GT_PATH)

    # pred_coo = getCoord(PRED_PATH)

    print(gt_coo)
    # print(iou(box2=pred_coo[0],box1=gt_coo[0]))
    img = drawBB(img,gt_coo,255,format=fm)
    # img = drawBB(img,pred_coo,0,format=fm)

    cv2.imshow('bb', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # main()
    a = torch.randn(3,5)
    b = torch.arange(0,15)
    print(a)
    print(b)

    a.copy_(b)
    print(a)
    # print(np.count_nonzero((a-b)==0))

    # print(np.where(a==0,a-1,a))



    # print(preprocessing.normalize(a*np.log(900/b),norm='l2'))
