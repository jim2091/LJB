# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:58:04 2021

@author: LJB
"""

from PIL import Image
import numpy as np
import pandas as pd
import scipy.linalg as la
import sys

mod = sys.modules[__name__]

im_num=20
dim=3

for i in range(dim):
    
    dataframe = pd.DataFrame(columns=range(1800))
    
    for k in range(im_num):
        im = Image.open('{}.png'.format(k+1))
        pix = np.array(im)
        dt = np.reshape(pix[:,:,i], (1, 1800))
        dt = pd.DataFrame(dt)
        dataframe = dataframe.append([dt])
        
    setattr(mod, 'df_{}'.format(i), dataframe)


    
    cov_matrix = np.cov(getattr(mod, 'df_{}'.format(i)).astype(float).transpose())
    eig_vals, eig_vecs = la.eig(cov_matrix)
    idx = eig_vals.argsort()[::-1]   
    setattr(mod, 'eig_vals_{}'.format(i), eig_vals[idx])
    setattr(mod, 'eig_vecs_{}'.format(i), eig_vecs[:,idx])


# eigface
a = np.zeros((45,40,4))
a = a.astype(int)
b = np.full((45,40), 255)


for i in range(im_num):
    
    for j in range(dim):
        v_i = getattr(mod, 'eig_vecs_{}'.format(j))
        v_i = v_i[:,i]
        face = np.abs(np.reshape(v_i, (45, 40)))
        face = (face/np.max(face))*255
        setattr(mod, 'face_{}'.format(j), face)
        
    a[:,:,0] = face_0
    a[:,:,1] = face_1
    a[:,:,2] = face_2
    a[:,:,3] = b
    eigface_i = Image.fromarray(a.astype('uint8'))
    eigface_i.save("eigface_{}.png".format(i+1))
