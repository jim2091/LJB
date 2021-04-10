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
    im = Image.open('{}_gray.png'.format(k+1))
    
    pix = np.array(im)
    dt = np.reshape(pix[:,:,0], (1, 1800))
    dt = pd.DataFrame(dt)
    
    dataframe = dataframe.append([dt])

#평균얼굴. eigenface와는 무슨 관련인지 사실 잘 모르겠음.
face_mean = np.mean(dataframe)

face_mean = np.array(face_mean).reshape((45,40,1))
b = np.full((45,40), 255)

a = np.zeros((45,40,4))
a[:,:,:] = face_mean
a[:,:,3] = b
a = a.astype(np.int64)

plt.imshow(a)
plt.axis('off')
plt.show

