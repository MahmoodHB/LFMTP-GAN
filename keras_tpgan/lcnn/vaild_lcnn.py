from keras.models import load_model 
import cv2
import numpy as np
import os
from skimage import img_as_float
from keras import backend as K
import tensorflow as tf
import pickle

DATASET_TYPE = 'CHI'
#DATASET_TYPE = 'FEI'

# In[] : reshape

def reshape_3d(img):
    size = img.shape
    img_3d = np.reshape(img, (1,size[0] ,size[1] ,1))
    
    return img_3d

# In[]: uint8 to float
    
def img2float(img):
    out_img = img_as_float(img)
    return out_img

# In[]: translate input

def img2tensor(img_dir):
    test_img = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
    test_img_3d = reshape_3d(test_img)
    test_tensor = img2float(test_img_3d) 
    return test_tensor

# In[] : bulid lcnn
    
def build_lcnn(model_path):
    lcnn = load_model(model_path)
    lcnn.summary()
    return lcnn

# In[] :  predict data

def lcnn_pred(true_tensor_dir ,pred_tensor_dir):
    true_tensor = img2tensor(true_tensor_dir)
    vec_true ,map_true = lcnn.predict(true_tensor,batch_size=1)
    
    pred_tensor = img2tensor(pred_tensor_dir)
    vec_pred ,map_pred = lcnn.predict(pred_tensor,batch_size=1)
    
    loss_val =K.mean(K.abs(vec_pred - vec_true)) + K.mean(K.abs(map_pred - map_true))
    
    show_tensor = tf.Session() 
    distence = show_tensor.run(loss_val)
    
    return distence
# In[] :  import front path in FEI

def FEI_front_img(img_path):
    front_path = img_path[:-2]+'11.jpg'
    
    return front_path

# In[] :  import front path in chinese dataset

def CHI_front_img(img_path):
    front_path = img_path[:-3]+'0.jpg'
    
    return front_path

# In[] : main function
    
if __name__ == "__main__":
    
    if DATASET_TYPE == 'CHI':
        datalist_path = 'E:/tpgan_keras(for FEI)/keras_tpgan/datalist/datalist_new.pkl'
        model_path = 'E:/tpgan_keras(for chinese)/lcnn/epoch1500_lr0.00100_loss0.122_valacc1.000.hdf5'
        dataset_dir = 'E:/tpgan_keras(for chinese)/crop_img(gray_jpg)/'
    elif DATASET_TYPE == 'FEI':
        datalist_path = 'E:/lcnn(for FEI)/datalist/datalist_test(Side).pkl'
        model_path = 'E:/tpgan_keras(for FEI)/lcnn/epoch1210_lr0.00100_loss0.295_valacc1.000.hdf5'
        dataset_dir = 'E:/tpgan_keras(for FEI)/FEI dataset(crop_img)/'
    else:
        print("You need to add some infomation in code.")
    
    
    sum_loss_val=0
    count=0
    lambda_ip=1 
    #lambda_ip=1e-3
    
    #build_lcnn_model
    lcnn = build_lcnn(model_path)
    
    if tf.gfile.Exists(datalist_path):
        with open(datalist_path, 'rb') as f:
            datalist = pickle.load(f)
        
            if DATASET_TYPE == 'CHI':
                
                for i, data_path in enumerate(datalist):
                    true_img_path = os.path.join(dataset_dir ,data_path)
                    pred_path = CHI_front_img(data_path)
                    pred_img_path = os.path.join(dataset_dir ,pred_path)
                    
                    
                    true_img_path = true_img_path + ".jpg"
                    sum_loss_val = sum_loss_val + lcnn_pred(true_img_path ,pred_img_path)
                    
                    count = count + 1
                    
            elif DATASET_TYPE == 'FEI':
                
                for i, data_path in enumerate(datalist):
                    true_img_path = os.path.join(dataset_dir ,data_path)
                    pred_path = FEI_front_img(data_path)
                    pred_img_path = os.path.join(dataset_dir ,pred_path)
                    
                    
                    true_img_path = true_img_path + ".jpg"
                    sum_loss_val = sum_loss_val + lcnn_pred(true_img_path ,pred_img_path)
                    
                    count = count + 1

    else:
        print("The datalist is not exist.")
    
    average_loss_val = sum_loss_val / count
    
    print("average distance : ", average_loss_val)
    

    

    
