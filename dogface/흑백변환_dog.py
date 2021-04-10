# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:11:27 2021

@author: LJB
"""

from PIL import Image

im_num=40

for k in range(im_num):
    im = Image.open('{}.png'.format(k+1))
    pix = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            L=0.2126*pix[i,j][0]+0.7152*pix[i,j][1]+0.0722*pix[i,j][2]
            L=int(L)
            pix[i,j]=(L,L,L)
    im.save('{}_gray.png'.format(k+1)) 
