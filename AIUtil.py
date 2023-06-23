import cv2
import os
import shutil
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

IMAGES_CAPTURED = 500


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('User'):
    os.makedirs('User')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')
if f'User.csv' not in os.listdir('User'):
    with open(f'User/User.csv','w') as f:
        f.write('Name,Roll')

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def add_user(username, userid):
    df = pd.read_csv(f'User/User.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'User/User.csv','a') as f:
            f.write(f'\n{username},{userid}')

def del_user(userid):
    df = pd.read_csv(f'User/User.csv', dtype={"Roll": str})
    username = df.loc[df['Roll'] == userid, 'Name'].iloc[0]
    flag = len(df) - 1
    df = df[df['Roll'] != userid]
    df.to_csv('User/User.csv', index=False)
    shutil.rmtree(f'static/faces/{username}_{userid}')
    print(f'Da xoa thanh cong {username}_{userid}')

    if flag != 0:
        train_model()
    else:
        os.remove('static/face_recognition_model.pkl')
    return 0

def edit_user(userid, new_username):
    df = pd.read_csv(f'User/User.csv', dtype={"Roll": str})
    old_username = df.loc[df['Roll'] == userid, 'Name'].iloc[0]
    df.loc[df['Roll'] == userid, 'Name'] = new_username
    df.to_csv('User/User.csv', index=False)

    old_folder_name = f'static/faces/{old_username}_{userid}'
    new_folder_name = f'static/faces/{new_username}_{userid}'
    os.rename(old_folder_name, new_folder_name)
    train_model()
    return 0

def extract_user():
    df = pd.read_csv(f'User/User.csv', dtype={"Roll": str})
    names = df['Name']
    rolls = df['Roll']
    l = len(df)
    return names,rolls,l

################## FUNCTIONS #########################

def home():
    names,rolls,times,l = extract_attendance()   
    data = GrandData(names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 
    return data
 
def homeUser():
    names,rolls,l = extract_user()   
    data = GrandData(names=names,rolls=rolls,l=l,totalreg=totalreg(),datetoday2=datetoday2) 
    return data

#### This function will run when we click on Take Attendance Button
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return GrandData(totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            add_attendance(identified_person)
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()    
    return GrandData(names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we add a new user
def add(userName, id):
    newusername = userName
    newuserid = str(id)
    userimagefolder = 'static/faces/'+newusername+'_'+newuserid
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/{IMAGES_CAPTURED//10}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j == IMAGES_CAPTURED:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    add_user(username=userName, userid=id)  # save to csv file
    names, rolls, l = extract_user()   # update list
    return GrandData(names=names,rolls=rolls,l=l,totalreg=totalreg(),datetoday2=datetoday2)


class GrandData:
    def __init__(self, totalreg, datetoday2, names, rolls,l, mess=None, times=None):
        self.totalreg = totalreg
        self.datetoday2 = datetoday2
        self.mess = mess
        self.names = names
        self.rolls = rolls
        self.times = times
        self.l = l
        


