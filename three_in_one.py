import os
import cv2
import face_alignment #exract
import pickle  #save data
import numpy as np

src_dataset_dir = 'D:/desktop/tpgan_keras/dataset_png'   # dataset
dest_dataset_dir = 'D:/desktop/tpgan_keras/landmark'     # exract landmark
jpg_dataset_dir = 'D:/desktop/tpgan_keras/date_jpeg'     # png2jpeg
out_dir = 'D:/desktop/tpgan_keras'                       # landmark in one


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#curdir = os.getcwd()  #print working pathway    
count={} #count    
out_dict = {} #landmark in one    
os.chdir(src_dataset_dir)  #change working pathway        
sessions = os.listdir('.')    
k=0    
for session in sessions:  #讀取角度
    #change pathway
    os.chdir(src_dataset_dir)
    print(session)  
    
    #count
    i=0
    # landmark in one
    out_dict[session] = {}
    
    # exract landmark
    subjects = os.listdir(session)
    out_session_dir = os.path.join(dest_dataset_dir,session)
    os.makedirs(out_session_dir, exist_ok=True) #create folder
    src_session_dir = os.path.join(src_dataset_dir,session)
   
    # png2jpeg
    out_jpg_session_dir = os.path.join(jpg_dataset_dir,session)
    os.makedirs(out_jpg_session_dir, exist_ok=True)
       
    for subject in subjects:  #讀取正反面
        #change pathway
        os.chdir(src_session_dir)
        print(session + " " + subject)
        
        # landmark in one
        out_dict[session][subject] = {}
        
        # exract landmark
        numbers = os.listdir(subject)
        out_subject_dir = os.path.join(out_session_dir,subject)
        os.makedirs(out_subject_dir, exist_ok=True) #create folder
        src_subject_dir = os.path.join(src_session_dir,subject)
        
        # png2jpeg
        out_jpg_subject_dir = os.path.join(out_jpg_session_dir,subject)
        os.makedirs(out_jpg_subject_dir, exist_ok=True)
        
    
        for number in numbers:  #讀取組別
            #change pathway
            os.chdir(src_subject_dir)
            print(session + " " + subject + " " +number)
            
            # landmark in one
            out_dict[session][subject][number] = {}
            
            # exract landmark
            out_number_dir = os.path.join(out_subject_dir,number)
            images = os.listdir(number)
            src_number_dir = os.path.join(src_subject_dir,number)
            os.makedirs(out_number_dir, exist_ok=True) #create folder
            
            #png2jpeg
            out_jpg_number_dir = os.path.join(out_jpg_subject_dir,number)
            os.makedirs(out_jpg_number_dir, exist_ok=True)
            
            for image in images: #讀取圖片
                i=i+1
                #change pathway
                os.chdir(src_number_dir)
                print(session + " " + subject + " " + number + " " + image)
                
                out_image = os.path.join(out_number_dir, os.path.splitext(image)[0] + '.pkl')
                
                # png2jpeg
                src_image = image[:-4]
                src_jpg_path = os.path.join(src_number_dir, src_image + '.png')
                dest_jpg_path = os.path.join(out_jpg_number_dir, src_image + '.jpg')
                cv2.imwrite(dest_jpg_path, cv2.imread(src_jpg_path))
                
                    
                # exract landmark
                img_path =os.path.join(src_number_dir,image)
                img =cv2.imread(src_jpg_path)

                
                landmarks =fa.get_landmarks(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                
                
                if landmarks is None:
                    print("No face detected: {}".format(img_path))
                    continue
                
                #for i in range(landmarks):
                    #print('landmarks(i)',landmarks[0])
                    
                with open(out_image, 'wb') as f:        
                    pickle.dump(landmarks[0], f)
                
                # landmark in one
                with open(out_image, 'rb') as f:
                    landmark_mat = pickle.load(f)
               
                #landmark_mat[i][0] [i][1]
                out_dict[session][subject][number][src_image] = landmark_mat.astype(np.uint16)

    count[session] = i
# landmark in one                
os.chdir(out_dir)

with open('landmarks.pkl', 'wb') as f:
    pickle.dump(out_dict, f)  #save pkl
    

print("count: ",count)

    