import cv2
import mediapipe as mp
import math as m
import numpy as np
from tkinter import Y

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model

def straightline(p,q,r):
    p1=np.array(p)
    q1=np.array(q)
    r1=np.array(r)
    f1 = p1-q1 # normalization of vectors
    e1 = q1-r1 # normalization of vectors
    angle = np.dot(f1, e1)
    angle1=m.acos(angle)
   
    if (m.pi-(m.pi/18))<=angle1<=(m.pi+(m.pi/18)):
        return True
    else:
        return False
def parallelline(p,q,r,s):          #pq rs quad  
    p1=np.array(p)
    q1=np.array(q)
    r1=np.array(r)
    s1=np.array(s)
    f1 = p1-q1 # normalization of vectors
    e1 = q1-r1 # normalization of vectors
    
    f2 = q1-r1 # normalization of vectors
    e2 = r1-s1 # normalization of vectors
    
    angle1 = np.dot(f1, e1) # calculates dot product 
    angle2 = np.dot(f2, e2)
    if (m.pi-(m.pi/18))<=m.acos(angle1)+m.acos(angle2)<=(m.pi+(m.pi/18)) or -(m.pi/18)<=m.acos(angle1)+m.acos(angle2)<=m.pi/18:
        return True  # calculated angle in radians to degree 
    else:
        return False
        
def angle(p,q,r):        
        p1=np.array(p)
        q1=np.array(q)
        r1=np.array(r)
        f1 = p1-q1 # normalization of vectors
        e1 = q1-r1 # normalization of vectors
        angle = np.dot(f1, e1)
        angle1=m.acos((angle))
                
        if 0<angle1<m.pi/2:
            return angle1,'a'
        elif angle1==m.pi/2:
            return angle1,'r'
        elif  angle1<m.pi:
            return angle1,'o'
        elif angle1==m.pi:
            return angle1,'s'
        else:
            return angle1,"c"
        
        
def checkAccuracy(results,yoga):
      
   if results.pose_landmarks:
       # right leg
       rl1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z]
       rl2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z]
       rl3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z]
       
       # left leg
       ll1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z]
       ll2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z]
       ll3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z]
       
       # right arm
       ra1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]
       ra2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z]
       ra3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z]
       
       # left arm
       la1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
       la2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z]
       la3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z]
       
       # mouth
       lm1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].z]
       rm1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].z]
       
       # ear
       le1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].z]
       re1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].z]
       
       
       accuracy=0
       
       if yoga=="adho mukha svanasana":
             if parallelline(rl2,rl1,ll1,ll2):
                accuracy+=0.15
             if parallelline(ra2,ra1,la1,la2):
                accuracy+=0.15
             if straightline(rl1,rl2,rl3):
                 accuracy+=0.135
             if straightline(ll1,ll2,ll3):
                 accuracy+=0.135
             if straightline(ra1,ra2,ra3):
                 accuracy+=0.135    
             if straightline(la1,la2,la3):
                 accuracy+=0.135
             if  (angle(ra1,rl1,rl2)[1]==angle(la1,ll1,ll2)[1]=="a" and angle(ra1,rl1,rl2)[0]==angle(la1,ll1,ll2)[0]) or (angle(ra1,rl1,rl2)[1]==angle(la1,ll1,ll2)[1]=="r" and angle(ra1,rl1,rl2)[0]==angle(la1,ll1,ll2)[0]):
                 accuracy+=0.15
             return accuracy
         
       elif yoga=="adho mukha vriksasana":
             if parallelline(rl2,rl1,ll1,ll2):
                accuracy+=0.15
             if parallelline(ra2,ra1,la1,la2):
                accuracy+=0.15
             if straightline(rl1,rl2,rl3):
                 accuracy+=0.135
             if straightline(ll1,ll2,ll3):
                 accuracy+=0.135
             if straightline(ra1,ra2,ra3):
                 accuracy+=0.135    
             if straightline(la1,la2,la3):
                 accuracy+=0.135
             if  straightline(ra1,rl1,rl2) and straightline(la1,ll1,ll2):
                 accuracy+=0.15
             return accuracy     
         
       elif yoga=="agnistambhasana":
             if parallelline(ll3,rl2,ll2,ll1):
                accuracy+=0.15
             if ( angle(ra1,rl1,rl2)[1]=="a" or angle(ra1,rl1,rl2)[1]=="r") and ( angle(ra1,rl1,rl2)[0]==angle(la1,ll1,ll2)[0] ):
                 accuracy+=0.15
             if  angle(ll1,ll2,ll3)[1]=="a" :
                 accuracy+=0.35
             if  angle(rl1,rl2,rl3)[1]=="a" :
                 accuracy+=0.35
             return accuracy      
         
       elif yoga=="akarna dhanurasana":
             if ( straightline(rl1,rl2,rl3) and angle(ll1,ll2,ll3)[1]=="a") or ( straightline(ll1,ll2,ll3) and angle(rl1,rl2,rl3)[1]=="a"):
                accuracy+=0.45
             if ( straightline(ra1,ra2,ra3) and angle(la1,la2,la3)[1]=="a") or ( straightline(la1,la2,la3) and angle(ra1,ra2,ra3)[1]=="a"):
                accuracy+=0.45
             if  m.dist(ra3,rl3)<=m.dist(lm1,rm1)*1.5:
                 accuracy+=0.05
             if  m.dist(la3,ll3)<=m.dist(lm1,rm1)*1.5:
                 accuracy+=0.05
                 
             return accuracy
         
       elif yoga=="ananda balasana":
           #   moooooorrrreeeeee    cheeeeeeeck
             if parallelline(rl3,rl2,ll2,ll3):
                accuracy+=0.15
             if  (angle(rl1,rl2,rl3)[1]=="o" or angle(rl1,rl2,rl3)[1]=="r") and (angle(rl1,rl2,rl3)[0]==angle(ll1,ll2,ll3)[0]):
                 accuracy+=0.15
             if  m.dist(ra3,rl3)<=m.dist(lm1,rm1)*1.5:
                 accuracy+=0.05
             if  m.dist(la3,ll3)<=m.dist(lm1,rm1)*1.5:
                 accuracy+=0.05
                 
             return accuracy  
            
       elif yoga=="anantasana":
             if  (m.dist(ra3,rl3)<=m.dist(lm1,rm1)*1.5 and m.dist(le1,la3)<=m.dist(lm1,rm1)*1.3 ) or (m.dist(la3,ll3)<=m.dist(lm1,rm1)*1.5 and m.dist(re1,ra3)<=m.dist(lm1,rm1)*1.3 ) :
                 accuracy+=0.5
             if angle(rl2,rl1,ll1)[1]=="r"  or  angle(ll2,ll1,rl1)[1]=="r":
                 accuracy+=0.25
             if (straightline(rl1,rl2,rl3) and straightline(ll1,rl1,rl2))  or (straightline(ll1,ll2,ll3) and straightline(rl1,ll1,ll2)):
                 accuracy+=0.25
             return accuracy
       
       elif yoga=="anjaneyasana":
             ########  cheeeck moooooooorrrrrrrrrrrreeeeeeeeeee
             if  (m.dist(ra3,la3)<=m.dist(lm1,rm1)*1.2 ):
                 accuracy+=0.2    
             if (( angle(rl1,rl2,rl3)[1]=="a" or angle(rl1,rl2,rl3)[1]=="r") and angle(ll1,ll2,ll3)[1]=="o") or  (((angle(ll1,ll2,ll3)[1]=="a" or angle(ll1,ll2,ll3)[1]=="r")and angle(rl1,rl2,rl3)[1]=="o")):
                 accuracy+=0.5
             if (straightline(ra1,ra2,ra3) and straightline(la1,la2,la3)):
                 accuracy+=0.15
             return accuracy
         
       elif yoga=="ardha bhekasana":             
             if  (m.dist(ra3,rl3)<=m.dist(lm1,rm1)*1.2) or (m.dist(la3,ll3)<=m.dist(lm1,rm1)*1.2):
                 accuracy+=0.25
             if ( angle(rl1,rl2,rl3)[1]=="a" and (angle(ll1,ll2,ll3)[1]=="o" or angle(ll1,ll2,ll3)[1]=="s")) or  ((angle(ll1,ll2,ll3)[1]=="a" and (angle(rl1,rl2,rl3)[1]=="o" or angle(rl1,rl2,rl3)[1]=="s"))):
                 accuracy+=0.5                 
             if angle(la1,ll1,ll2)[1]==angle(ra1,rl1,rl2)[1]=="o":
                 accuracy+=0.25
             return accuracy     
    
       elif yoga=="ardha chandrasana":
             if  straightline(ra1,ra2,ra3) and straightline(la1,la2,la3) and straightline(la2,la1,ra1) and straightline(ra2,ra1,la1):
                 accuracy+=0.25                 
             if straightline(rl1,rl2,rl3) and straightline(ll1,ll2,ll3):
                 accuracy+=0.25     
             if angle(ll2,ll1,rl1)[1]=="r" or angle(rl2,rl1,ll1)[1]=="r":
                 accuracy+=0.25
             if parallelline(ra2,ra1,rl1,rl2) or parallelline(la2,la1,ll1,ll2):
                accuracy+=0.25
             return accuracy      
         
       elif yoga=="ardha matsyendrasana":
             if  straightline(ra1,ra2,ra3) and straightline(la1,la2,la3) and straightline(la2,la1,ra1) and straightline(ra2,ra1,la1):
                 accuracy+=0.25                 
             if straightline(rl1,rl2,rl3) and straightline(ll1,ll2,ll3):
                 accuracy+=0.25     
             if angle(ll2,ll1,rl1)[1]=="r" or angle(rl2,rl1,ll1)[1]=="r":
                 accuracy+=0.25
             if parallelline(ra2,ra1,rl1,rl2) or parallelline(la2,la1,ll1,ll2):
                accuracy+=0.25
             return accuracy      
         
            
         
            
         
            
         
            
         
            
         
            
         
            
         
            
       
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# For webcam input:
vid = cv2.VideoCapture("video.mp4")
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
  ret, image = vid.read()
  yoga="adho mukha svanasana"
  while ret :
    ret, image = vid.read()
    
    if ret==False:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    
    
    acc=checkAccuracy(results,yoga)                  ##############################   
    
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    #image=cv2.flip(image, 1)
    if acc!=None:
        if acc>=0.5:
             print("!!!!!!!!!!!!!!!!!!!!!!!!check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        cv2.putText(image,str(acc*100)+"%",( int(image.shape[0]/2) ,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    cv2.imshow('MediaPipe Pose', image)
    #print(mp_pose.POSE_CONNECTIONS)
    
    
     
    
    
    if cv2.waitKey(5) == ord(' '):
        print("done")
        break
    elif cv2.waitKey(5) == ord('c'):
        cv2.imwrite("test.jpg",image)
        image=load_img("test.jpg",1,target_size=(150,150)) 
        img=img_to_array(image)
        img=np.expand_dims(img,axis=0)

        ####################################
        m1=load_model("Modals/ALL(1).h5")
        #print(m1.summary())
        pred=m1(img)
        pred=np.array(pred)
        pred=pred.argmax()
        pred=int(pred)
        yoga1={0: 'adho mukha svanasana', 1: 'adho mukha vriksasana', 2: 'agnistambhasana', 3: 'akarna dhanurasana', 4: 'ananda balasana', 5: 'anantasana', 6: 'anjaneyasana', 7: 'ardha bhekasana', 8: 'ardha chandrasana', 9: 'ardha matsyendrasana', 10: 'ardha pincha mayurasana', 11: 'ardha uttanasana', 12: 'ashtanga namaskara', 13: 'astavakrasana', 14: 'baddha konasana', 15: 'bakasana', 16: 'balasana', 17: 'bhairavasana', 18: 'bharadvajasana i', 19: 'bhekasana', 20: 'bhujangasana', 21: 'bhujapidasana', 22: 'bitilasana', 23: 'camatkarasana', 24: 'chakravakasana', 25: 'chaturanga dandasana', 26: 'dandasana', 27: 'dhanurasana', 28: 'durvasasana', 29: 'dwi pada viparita dandasana', 30: 'eka pada koundinyanasana i', 31: 'eka pada koundinyanasana ii', 32: 'eka pada rajakapotasana', 33: 'eka pada rajakapotasana ii', 34: 'ganda bherundasana', 35: 'garbha pindasana', 36: 'garudasana', 37: 'gomukhasana', 38: 'halasana', 39: 'hanumanasana', 40: 'janu sirsasana', 41: 'jathara parivartanasana', 42: 'kapotasana', 43: 'karnapidasana', 44: 'krounchasana', 45: 'kukkutasana', 46: 'kurmasana', 47: 'lolasana', 48: 'makara adho mukha svanasana', 49: 'makarasana', 50: 'malasana', 51: 'marichyasana i', 52: 'marichyasana iii', 53: 'marjaryasana', 54: 'matsyasana', 55: 'mayurasana', 56: 'natarajasana', 57: 'navasana', 58: 'padangusthasana', 59: 'padmasana', 60: 'parighasana', 61: 'paripurna navasana', 62: 'parivrtta janu sirsasana', 63: 'parivrtta parsvakonasana', 64: 'parivrtta trikonasana', 65: 'parsva bakasana', 66: 'parsvottanasana', 67: 'pasasana', 68: 'paschimottanasana', 69: 'phalakasana', 70: 'pincha mayurasana', 71: 'prasarita padottanasana', 72: 'purvottanasana', 73: 'rajakapotasana', 74: 'salabhasana', 75: 'salamba bhujangasana', 76: 'salamba sarvangasana', 77: 'salamba sirsasana', 78: 'sarvangasana', 79: 'savasana', 80: 'setu bandha sarvangasana', 81: 'simhasana', 82: 'sukhasana', 83: 'supta baddha konasana', 84: 'supta matsyendrasana', 85: 'supta padangusthasana', 86: 'supta virasana', 87: 'tadasana', 88: 'tittibhasana', 89: 'tolasana', 90: 'tulasana', 91: 'upavistha konasana', 92: 'urdhva dhanurasana', 93: 'urdhva hastasana', 94: 'urdhva mukha svanasana', 95: 'urdhva prasarita eka padasana', 96: 'ustrasana', 97: 'utkatasana', 98: 'uttana shishosana', 99: 'uttanasana', 100: 'utthita ashwa sanchalanasana', 101: 'utthita hasta padangustasana', 102: 'utthita parsvakonasana', 103: 'utthita trikonasana', 104: 'vajrasana', 105: 'vasisthasana', 106: 'viparita dandasana', 107: 'viparita karani', 108: 'virabhadrasana i', 109: 'virabhadrasana ii', 110: 'virabhadrasana iii', 111: 'virasana', 112: 'vriksasana', 113: 'vrischikasana', 114: 'yoganidrasana'}
        yoga=yoga1[pred]
        print(yoga1[pred])
        
      
vid.release()
cv2.destroyAllWindows() 
print("done")