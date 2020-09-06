#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# In[2]:


path='C:\\Users\\ASUS\\Attendance face recog\\Photos'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# In[3]:


def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodelistknown=findEncodings(images)
print("Encoding Complete")


# In[4]:


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myData=f.readlines()
        nameList=[]
        for line in myData:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            mystring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{mystring}') 


# In[5]:


cap=cv2.VideoCapture(0)


# In[6]:


while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    facescurr=face_recognition.face_locations(imgS)
    encodecurr=face_recognition.face_encodings(imgS,facescurr)
    for encodeface,faceloc in zip(encodecurr,facescurr):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        facedis=face_recognition.face_distance(encodelistknown,encodeface)
        print (facedis)
        matchIndex=np.argmin(facedis)
        
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


# In[ ]:





# In[ ]:




