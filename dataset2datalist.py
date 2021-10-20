import os
import pickle
import pandas as pd


src_dataset_dir = 'D:/desktop/tpgan_keras/test_data' 
#src_dataset_dir = 'D:/desktop/tpgan_keras/dataset_png' 
dest_dataset_dir = 'D:/desktop/tpgan_keras'
#dest_dataset_dir = 'D:/desktop/tpgan_keras/out_data'

os.chdir(src_dataset_dir) #change pathway

sessions = os.listdir('.')
#   session, subject, number, image
  
out_dict = []

for session in sessions:
    os.chdir(src_dataset_dir)
    
    print(session)
  
    subjects = os.listdir(session)
    
    for subject in subjects:
        if subject is 't': #將正面移除
            continue
        src_session_dir = os.path.join(src_dataset_dir, session)
        os.chdir(src_session_dir)
        
        print(session + "  " + subject)            
        
        numbers = os.listdir(subject)
       
        for number in numbers:
            src_subject_dir = os.path.join(src_session_dir, subject)
            os.chdir(src_subject_dir)
            
            print(session + "  " + subject+ " " + number) 
            
            images = os.listdir(number)
            
            for image in images:
                src_number_dir = os.path.join(src_subject_dir, number)
                os.chdir(src_number_dir)
                print(session + "  " + subject + " " + number + " " + image)
                
                data_name = os.path.basename(image)[:-4]
                data_path = os.path.join( session, subject, number, data_name)
                out_dict.append(data_path)
        
os.chdir(dest_dataset_dir)

#df = pd.DataFrame(out_dict)
#out_datalist = df.values.tolist()

with open('datalist_test.pkl', 'wb') as f:
    pickle.dump(out_dict, f)  #save pkl