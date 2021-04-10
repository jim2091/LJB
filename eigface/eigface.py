# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:58:04 2021

@author: LJB
"""

from PIL import Image
import numpy as np
import pandas as pd
import scipy.linalg as la

dataframe = pd.DataFrame(columns=range(1800))

im_num=20

for k in range(im_num):
    im = Image.open('{}_gray.png'.format(k+1))
    pix = np.array(im)
    dt = np.reshape(pix[:,:,0], (1, 1800))
    dt = pd.DataFrame(dt)
    
    dataframe = dataframe.append([dt])

cov_matrix = np.cov(dataframe.astype(float).transpose())

eig_vals, eig_vecs = la.eig(cov_matrix)
idx = eig_vals.argsort()[::-1]   
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:,idx]

# eigface

a = np.zeros((45,40,4))
a = a.astype(int)
b = np.full((45,40), 255)

for i in range(im_num):
    v_i = eig_vecs[:,i]
    face_i = np.abs(np.reshape(v_i, (45, 40)))
    face_i = (face_i/np.max(face_i))*255
    face_i = np.array(face_i).reshape((45,40,1))
    a[:,:,:] = face_i
    a[:,:,3] = b
    
    eigface_i = Image.fromarray(a.astype('uint8'))
    eigface_i.save("eigface_{}.png".format(i+1))

#reconstruction
k_list = [2, 5, 10, 20]
for k in k_list:
     e_k = eig_vecs[:,0:k]
     c_k = np.matmul(np.array(dataframe), e_k)
     x_ = np.matmul(e_k, c_k.T)
     
     for j in range(20):
         face_j = np.abs(np.reshape(x_[:,j], (45,40)))
         face_j = (face_j/np.max(face_j))*255
         face_j = np.array(face_j).reshape((45,40,1))
         a[:,:,:] = face_j
         a[:,:,3] = b         
         eigface_j = Image.fromarray(a.astype('uint8'))
         eigface_j.save("{}_k{}.png".format(j+1, k))
         
