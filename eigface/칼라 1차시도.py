# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:58:04 2021

@author: LJB
"""


###########일단 1차시도. 코드가 돌아가긴 돌아감. 근데 결과가 거의 노이즈. 어디서 잘못된건지 추적 요망.
###########원하던 대로 각 층별로 데이터 뽑고, cov만들고, eig만들고, 층별로 만든걸 쌓아서 face 만드는거까지 문제 없어보임
###########그렇다면 애초에 방법이 잘못됐다? 층별로 계산하는 것이 아니라 전체를 한번에 하는 방법이 있는걸까

##애초에 그게 맞는거 같다. 흑백으로 성공한것도 매틀랩 코드로 만든거에 비해 명도가 전체적으로 높아졌던게
##매틀랩에서는 전체에 대해 한번에 계산해서 255부분들도 다 같이 들어있었기 때문이 아닐까.
##그건 아닌가.. 매틀랩에서는 애초에 흑백이미지를 45*40으로 받아왔다. 파이썬에서는 그런게 없나
##PIL말고 다른걸로 가져오면 어떤 형식으로 받아올 수 있는지 확인해보자.

from PIL import Image
import numpy as np
import pandas as pd
import scipy.linalg as la
import sys

mod = sys.modules[__name__]

im_num=20
dim=3

# 칼라사진의 각 층별 값 별도 저장.
for i in range(dim):
    
    dataframe = pd.DataFrame(columns=range(1800))
    
    for k in range(im_num):
        im = Image.open('{}.png'.format(k+1))
        pix = np.array(im)
        dt = np.reshape(pix[:,:,i], (1, 1800))
        dt = pd.DataFrame(dt)
        dataframe = dataframe.append([dt])
        
    setattr(mod, 'df_{}'.format(i), dataframe)


    # 현재 데이터들의 dtype이 object라서 오류발생. float으로 변환.
    cov_matrix = np.cov(getattr(mod, 'df_{}'.format(i)).astype(float).transpose())
    eig_vals, eig_vecs = la.eig(cov_matrix)
    idx = eig_vals.argsort()[::-1]   
    setattr(mod, 'eig_vals_{}'.format(i), eig_vals[idx])
    setattr(mod, 'eig_vecs_{}'.format(i), eig_vecs[:,idx])


# eigface 저장하기(20개까지)
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
    #원래 image 타입이 uint8인듯..ㅇㅇ
    eigface_i = Image.fromarray(a.astype('uint8'))
    eigface_i.save("eigface_{}.png".format(i+1))
#####일단 여기까지 색이 있는 얼굴모양으로 나오긴 했는데 아마 잘..나온거겠지?
#####재조합까지 해바야 확실할거 같은데 오늘은 여기까지...   


# #reconstruction
# k_list = [2, 5, 10, 20]
# for k in k_list:
#      e_k = eig_vecs[:,0:k]
#      c_k = np.matmul(np.array(dataframe), e_k)
#      x_ = np.matmul(e_k, c_k.T)
     
#      for j in range(20):
#          face_j = np.abs(np.reshape(x_[:,j], (45,40)))
#          face_j = (face_j/np.max(face_j))*255
#          face_j = np.array(face_j).reshape((45,40,1))
#          a[:,:,:] = face_j
#          a[:,:,3] = b         
#          eigface_j = Image.fromarray(a.astype('uint8'))
#          eigface_j.save("{}_k{}.png".format(j+1, k))
         
# #성공!