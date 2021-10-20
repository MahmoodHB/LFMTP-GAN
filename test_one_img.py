import os
import cv2

import pickle
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image

#display color
c=1

test_dir = 'D:/desktop/tpgan_keras/date_jpeg/15/f/01/201_01_01_050_06_cropped_test.jpg'
test_landmark_dir ='D:/desktop/tpgan_keras/landmark/15/f/01/201_01_01_050_06_cropped_test.pkl'

#display point
DISPLAY = True
#DISPLAY = False


#show face image
img = cv2.imread(test_dir)
img.setflags(write=1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
'''
plt.imshow(img)
plt.axis('off')
plt.show
'''
'''
cv2.imshow('Org Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#load landmark
with open(test_landmark_dir, 'rb') as f:
    landmarks = pickle.load(f)
    
min_x = 128
max_x = 0
    
landmark = landmarks
    
for i in range(len(landmark)): 
    #print('第',i,'組 -- y :',landmark[i][1],' x :',landmark[i][0])
    
    if DISPLAY == True:
        img[int(landmark[i][1]),int(landmark[i][0]),c] = 255
        img[int(landmark[i][1])-1,int(landmark[i][0]),c] = 255
        img[int(landmark[i][1]),int(landmark[i][0])-1,c] = 255
        img[int(landmark[i][1])-1,int(landmark[i][0])-1,c] = 255
        img[int(landmark[i][1])+1,int(landmark[i][0]),c] = 255
        img[int(landmark[i][1]),int(landmark[i][0])+1,c] = 255
        img[int(landmark[i][1])+1,int(landmark[i][0])+1,c] = 255
        img[int(landmark[i][1])-1,int(landmark[i][0])+1,c] = 255
        img[int(landmark[i][1])+1,int(landmark[i][0])-1,c] = 255
            
    if landmark[i][0] < 0 or landmark[i][0] > 128 :
        continue
    
'''
    if min_x > landmark[i][0]:
        min_x = landmark[i][0]

    if max_x < landmark[i][0]:
        max_x = landmark[i][0]
print(' | 最小x軸 :',min_x,' | 最大x軸 :',max_x)

crop_img = img[:,int(min_x):int(max_x)]
'''


#corp img




plt.imshow(img)
plt.axis('off')
plt.show 
 
#padding
#padding = [255,255,255]

#padding_crop_img=cv2.copyMakeBorder(crop_img,0,0,int(min_x),128-int(max_x),cv2.BORDER_CONSTANT,value=padding)

    
