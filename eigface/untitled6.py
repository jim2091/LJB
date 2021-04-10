# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:58:04 2021

@author: LJB
"""

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.DataFrame(columns=range(1800))

im_num=20

for k in range(im_num):
    im = Image.open('{}.gray.png'.format(k+1))
    
    pix = np.array(im)
    dt = np.reshape(pix[:,:,0], (1, 1800))
    dt = pd.DataFrame(dt)
    
    dataframe = dataframe.append([dt])

#평균 0으로 표준화,std로 나누는건 안해도 되려나
data_std = (dataframe - np.mean(dataframe, axis = 0)) / np.std(dataframe, axis = 0)
#np.cov에 맞게 행열전환
data_std = data_std.transpose()
#현재 데이터들의 dtype이 object라서 오류발생. float으로 변환.
cov_matrix = np.cov(data_std.astype(float))
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

u1 = eig_vecs[0]
u1 = np.reshape(u1, (1800,1))
x = data_std.transpose()
eig_face1 = np.dot(x, u1) + np.mean(dataframe, axis = 0)

'''
eig_face1 = np.dot(eig_vecs.T[0,:], data_std) + np.mean(dataframe, axis = 0)
eig_face1 = np.array(eig_face1).reshape((45,40,1))

b = np.full((45,40), 255)

a = np.zeros((45,40,4))
a[:,:,:] = eig_face1
a[:,:,3] = b
a = a.astype(np.uint64)

plt.imshow(a)
plt.axis('off')
plt.show
'''