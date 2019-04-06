import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalizeHist(img):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    claheImg = clahe.apply(img)
    # claheImg = cv2.equalizeHist(img)
    claheImg = cv2.fastNlMeansDenoising(claheImg,None,4,7,21)

    return claheImg

def showTensor(out):
    '''
    Visualize a Tensors
    input: a Tensors on cpu (1x1xhxw)
    '''
    out = out.type('torch.ByteTensor')
    out = out.cpu()
    out = out.detach().numpy()
    img = out[0].transpose((1, 2, 0))
    cv2.imshow('sss', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread('I01208.jpg',0)

    a = equalizeHist(img)

    cv2.imshow('raw', img)
    cv2.imshow('after', a)
    plt.hist(img.ravel(),256,[0,256]); plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
