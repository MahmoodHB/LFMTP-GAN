import os
import cv2
import face_alignment #exract

src_dataset_dir = 'E:/tpgan_keras/dataset_png'   # dataset

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#curdir = os.getcwd()  #print working pathway    
out_dict = {} #landmark in one    
os.chdir(src_dataset_dir)  #change working pathway        
sessions = os.listdir('.')   
 
landmark_count=0
landmark_leye=[0,0]
landmark_reye=[0,0]
landmark_nose=[0,0]
landmark_mouth=[0,0]

for session in sessions:  #讀取角度
    #change pathway
    os.chdir(src_dataset_dir)
    print(session)  
    
    #count
    # landmark in one
    out_dict[session] = {}
    
    # exract landmark
    subjects = os.listdir(session)
    src_session_dir = os.path.join(src_dataset_dir,session)
         
    for subject in subjects:  #讀取正反面
        #change pathway
       if subject == 't':
            os.chdir(src_session_dir)
            print(session + " " + subject)
             
            # exract landmark
            numbers = os.listdir(subject)
            src_subject_dir = os.path.join(src_session_dir,subject)
        
            for number in numbers:  #讀取組別
                #change pathway
                os.chdir(src_subject_dir)
                print(session + " " + subject + " " +number)
                               
                # exract landmark
                images = os.listdir(number)
                src_number_dir = os.path.join(src_subject_dir,number)
                
                for image in images: #讀取圖片
                    #change pathway
                    os.chdir(src_number_dir)
                    print(session + " " + subject + " " + number + " " + image)
                                    
                    # png2jpeg
                    src_image = image[:-4]
                    src_jpg_path = os.path.join(src_number_dir, src_image + '.png')
                   
                        
                    # exract landmark
                    img_path =os.path.join(src_number_dir,image)
                    img =cv2.imread(src_jpg_path)
    
                    
                    landmarks =fa.get_landmarks(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                    
                                       
                    if landmarks is None:
                        print("No face detected: {}".format(img_path))
                        continue
                    else:
                        landmark_count = landmark_count + 1
                        landmark_leye  = landmark_leye  + landmarks[0][42:48]
                        landmark_reye  = landmark_reye  + landmarks[0][36:42]
                        landmark_nose  = landmark_nose  + landmarks[0][31:36]
                        landmark_mouth = landmark_mouth + landmarks[0][48:60]
                        
average_landmark_leye =landmark_leye /landmark_count
average_landmark_reye =landmark_reye /landmark_count
average_landmark_nose =landmark_nose /landmark_count
average_landmark_mouth=landmark_mouth/landmark_count

total_average_landmark_leye = sum(average_landmark_leye) /len(average_landmark_leye)
total_average_landmark_reye = sum(average_landmark_reye) /len(average_landmark_reye)
total_average_landmark_nose = sum(average_landmark_nose) /len(average_landmark_nose)
total_average_landmark_mouth= sum(average_landmark_mouth)/len(average_landmark_mouth)

print("\naverage_landmark_leye  :", average_landmark_leye )
print("\naverage_landmark_reye  :", average_landmark_reye )
print("\naverage_landmark_nose  :", average_landmark_nose )
print("\naverage_landmark_mouth :", average_landmark_mouth)
print("\n------------------------------------------\n")

print("\ntotal_average_landmark_leye  :", total_average_landmark_leye )
print("\ntotal_average_landmark_reye  :", total_average_landmark_reye )
print("\ntotal_average_landmark_nose  :", total_average_landmark_nose )
print("\ntotal_average_landmark_mouth :", total_average_landmark_mouth)

    