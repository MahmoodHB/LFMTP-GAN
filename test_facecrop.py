import os
import cv2

import pickle
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from PIL import Image

#display color
c=1

src_dataset_dir = 'D:/desktop/tpgan_keras/dataset_png'
landmark_dir = 'D:/desktop/tpgan_keras/landmark'
our_dir = 'D:/desktop/tpgan_keras/crop_img'

#test_dir = 'D:/desktop/tpgan_keras/date_jpeg/15/f/01/201_01_01_050_06_cropped_test.jpg'
#test_landmark_dir ='D:/desktop/tpgan_keras/landmark/15/f/01/201_01_01_050_06_cropped_test.pkl'

#display point
#DISPLAY = True
DISPLAY = False


def crop_img(img_dir,landmark_dir):
    #show face image
    img = cv2.imread(img_dir)
    img.setflags(write=1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    with open(landmark_dir, 'rb') as f:
        landmark = pickle.load(f)
        
        min_x = 128
        max_x = 0
        
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
        
        if min_x > landmark[i][0]:
            min_x = landmark[i][0]
    
        if max_x < landmark[i][0]:
            max_x = landmark[i][0]
    
    print(' | 最小x軸 :',min_x,' | 最大x軸 :',max_x)
    
    #corp img
    crop_img = img[:,int(min_x):int(max_x)]
    

    '''
    plt.imshow(crop_img)
    plt.axis('off')
    plt.show 
    ''' 
    #padding
    padding = [255,255,255]
    
    padding_crop_img=cv2.copyMakeBorder(crop_img,0,0,int(min_x),128-int(max_x),cv2.BORDER_CONSTANT,value=padding)

    
    return padding_crop_img
    #return crop_img
    

if __name__ == '__main__':
    count={} #count     
    os.chdir(src_dataset_dir)  #change working pathway        
    sessions = os.listdir('.')    
      
    for session in sessions:  #讀取角度
        os.chdir(src_dataset_dir)
        print(session)       
        
        #count
        i=0
        
        # exract landmark
        subjects = os.listdir(session)
        out_session_dir = os.path.join(our_dir,session)
        os.makedirs(out_session_dir, exist_ok=True) #create folder
        landmark_session_dir = os.path.join(landmark_dir,session)
        src_session_dir = os.path.join(src_dataset_dir,session)     
           
        for subject in subjects:  #讀取正反面
            #change pathway
            os.chdir(src_session_dir)
            print(session + " " + subject)
            
            # exract landmark
            numbers = os.listdir(subject)
            out_subject_dir = os.path.join(out_session_dir,subject)
            os.makedirs(out_subject_dir, exist_ok=True) #create folder
            src_subject_dir = os.path.join(src_session_dir,subject)
            landmark_subject_dir = os.path.join(landmark_session_dir,subject)
        
            for number in numbers:  #讀取組別
                #change pathway
                os.chdir(src_subject_dir)
                print(session + " " + subject + " " +number)
                
                images = os.listdir(number)
                out_number_dir = os.path.join(out_subject_dir,number)
                os.makedirs(out_number_dir, exist_ok=True) #create folder
                src_number_dir = os.path.join(src_subject_dir,number)
                landmark_number_dir = os.path.join(landmark_subject_dir,number)
                
                for image in images: #讀取圖片
                    i=i+1
                    #change pathway
                    os.chdir(src_number_dir)
                    print(session + " " + subject + " " + number + " " + image,end='')
                    image_name = image[:-4]
                    src_image_dir = os.path.join(src_number_dir, image_name + '.png')
                    landmark_image_dir = os.path.join(landmark_number_dir, image_name + '.pkl')
                    
                    out_image_dir = os.path.join(out_number_dir, image_name + '.jpg')
                    crop_image = crop_img(src_image_dir,landmark_image_dir)
                    cv2.imwrite(out_image_dir ,crop_image)
                    
    
        count[session] = i
    
    print("count: ",count)