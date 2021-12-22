import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import math

def cut(input):
    (h,w) = input.shape[0:2]
    output = []
    ret1,th1 = cv2.threshold(input[:,:,0],230,255,cv2.THRESH_BINARY)
    ret2,th2 = cv2.threshold(input[:,:,1],230,255,cv2.THRESH_BINARY)
    ret3,th3 = cv2.threshold(input[:,:,2],230,255,cv2.THRESH_BINARY)
    output = th1 + th2 + th3   
    return output


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False       # 中文标题设置


img = cv2.imread('myself.jpg')
img = img[:, :, ::-1]
# 直方图均衡化增强
result1 = img.copy()
for i in range(3):
    result1[:,:,i] = cv2.equalizeHist(img[:,:,i])


# laplace算子增强
result2 = img.copy()
laplace = np.array([[0,-1,0],[0,5,0],[0,-1,0]])
result2 = cv2.filter2D(img,-1,laplace)


# 大津阈值分割
output1 = cut(result1)
output2 = cut(result2)

print(output2)


plt.figure(1)
plt.subplot(221)
plt.imshow(img)
plt.title('原图', fontsize = 18)
plt.subplot(222)
plt.imshow(result1)
plt.title('均衡化增强', fontsize = 18)
plt.subplot(223)
plt.imshow(result2)
plt.title('laplace增强', fontsize = 18)



plt.figure(2)
plt.subplot(121)
plt.imshow(output1,cmap='gray')
plt.title('均衡化增强后分割', fontsize = 18)
plt.subplot(122)
plt.imshow(output2,cmap='gray')
plt.title('laplace增强后分割', fontsize = 18)
plt.show()


