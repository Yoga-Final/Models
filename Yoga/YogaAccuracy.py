import cv2
import mediapipe as mp
import math as m
import numpy as np

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model

def lengthapart(p,q,r,s):    #r,s  shoulder coordinates   p,q part to 
     if m.dist(p,q)==m.dist(r,s):
         return 1
     elif m.dist(p,q)<m.dist(r,s):
         return m.dist(p,q)/m.dist(r,s)
     elif m.dist(p,q)<m.dist(r,s)*2:
         return m.dist(p,q)/m.dist(r,s)-1
     else:
         return 0
     
def parallelplane(p1,q1,r1,p2,q2,r2):     #  p1,q1,r1 one plane   ow
    p1=np.array(p1)
    q1=np.array(q1) 
    r1=np.array(r1)
    s1=[r1[2]*q1[1]-q1[2]*r1[1],p1[2]*r1[1]-r1[2]*p1[1],q1[2]*p1[1]-p1[2]*q1[1]]
    s1=np.array(s1)
    s1 = s1 / np.linalg.norm(s1)
    
    p2=np.array(p2)
    q2=np.array(q2) 
    r2=np.array(r2)
    s2=[r2[2]*q2[1]-q2[2]*r2[1],p2[2]*r2[1]-r2[2]*p2[1],q2[2]*p2[1]-p2[2]*q2[1]]
    s2=np.array(s2)
    s2 = s2 / np.linalg.norm(s2)
    
    angle1 = np.dot(s1, s2)
    angle1=m.acos((angle1))
    
    if angle1>m.pi/2:
        ratio=angle1/m.pi
        return ratio
    else:
        ratio=(m.pi-angle1)/m.pi
        return ratio

def touch(p,q,r,s):    #  p,q  touching points       r s   reference
    if m.dist(p,q)<=m.dist(r,s)*5:
        return 1
    elif m.dist(p,q)<=m.dist(r,s)*7:
        return 0.5
    else:
        return 0
    

def parallelline(p,q,r,s,angle):          #pq rs quad  
     p=np.array(p)
     q=np.array(q) 
     r=np.array(r)
     s=np.array(s)

     f = p-q  
     f1 = f / np.linalg.norm(f)
     e = s-r 
     e1 = e / np.linalg.norm(e)

     angle1 = np.dot(f1, e1)
     
     if angle=="ap":
         angle1=m.acos((angle1))
         ratio=angle1/m.pi
         return ratio
     else:
         angle1=m.acos((angle1))
         ratio=(m.pi-angle1)/m.pi
         return ratio
        
def angle(p,q,r,angle,ideal):        #([],[],[],"a",m.pi/6)   
        p=np.array(p)
        q=np.array(q) 
        r=np.array(r)

        f = p-q  
        f1 = f / np.linalg.norm(f)
        e = r-q 
        e1 = e / np.linalg.norm(e)

        angle1 = np.dot(f1, e1)

        angle1=m.acos((angle1))
                        
        if angle=='a':
            if angle1<(m.pi/2):
                if angle1>ideal:
                    ratio=1-(angle1-ideal)/ideal
                    return ratio,"y"
                else:
                    ratio=angle1/ideal
                    return ratio,"y"
                
            else:
                return 0.0,"n"
                
        elif angle=='r':
            ratio=angle1/(m.pi/2)
            if angle1==(m.pi/2):
                return ratio,"y"
            else:
                if angle1<(m.pi/2):
                    return ratio,"n"
                else:
                    ratio=1-(angle1-m.pi/2)/(m.pi/2)
                    return ratio,"n"
                
                
        elif  angle=='o':
            ratio=angle1/ideal
            if (m.pi/2)<angle1<(m.pi):
                if angle1>ideal:
                    ratio=1-(angle1-ideal)/ideal
                    return ratio,"y"
                else:
                    ratio=angle1/ideal
                    return ratio,"y"
            else:
                return 0.0,"n"
            
            
        elif angle=='s':
            ratio=angle1/m.pi
            if angle1==m.pi:
                return ratio,"y"
            else:
                return ratio,"n"
        
        
        
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
       
       # right hand
       rh1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].z]
       rh2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].z]
       rh3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].z]
       
       # LEFt hand
       lh1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].z]
       lh2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].z]
       lh3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].z]
       
       # nose
       n0=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z]
       
       # left feet
       lf1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z]
       lf2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].z]
       
       # right feet
       rf1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z]
       rf2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].z]
       
       
       accuracy=0
       
       if yoga=="adho mukha svanasana":
             accuracy+=0.15*parallelline(rl3,rl1,ll1,ll3,"p")    # feet hip aaprt
             accuracy+=0.15*parallelline(ra3,ra1,la1,la3,"p")    #  hands shoulded apart
             accuracy+=0.135*angle(rl1,rl2,rl3,"s",m.pi)[0]      #   right leg straight
             accuracy+=0.135*angle(ll1,ll2,ll3,"s",m.pi)[0]      #   left leg straight
             accuracy+=0.135*angle(ra1,ra2,ra3,"s",m.pi)[0]      #   right arm straight
             accuracy+=0.135*angle(la1,la2,la3,"s",m.pi)[0]      #   left arm straight
             if  (angle(ra1,rl1,rl2,"a",m.pi)[1]==angle(la1,ll1,ll2,"a",m.pi)[1]=="y" )or ( angle(ra1,rl1,rl2,"r",m.pi)[1]==angle(la1,ll1,ll2,"r",m.pi)[1]=="y"):
                 accuracy+=0.15
             return (accuracy)
        
       elif yoga=="adho mukha vriksasana":
             accuracy+=0.15*parallelline(rl2,rl1,ll1,ll2,"p")
             accuracy+=0.15*parallelline(ra2,ra1,la1,la2,"p")
             accuracy+=0.135*angle(rl1,rl2,rl3,"s",m.pi)[0]
             accuracy+=0.135*angle(ll1,ll2,ll3,"s",m.pi)[0]
             accuracy+=0.135*angle(ra1,ra2,ra3,"s",m.pi)[0]
             accuracy+=0.135*angle(la1,la2,la3,"s",m.pi)[0]
             if  angle(ra1,rl1,rl2,"s",m.pi)[1]==angle(la1,ll1,ll2,"s",m.pi)[1]=="y" or angle(ra1,rl1,rl2,"o",m.pi)[1]==angle(la1,ll1,ll2,"o",m.pi)[1]=="y":
                 accuracy+=0.15
             return accuracy     
         
      
       elif yoga=="akarna dhanurasana":
             if angle(ll1,ll2,ll3,"a",m.pi)[1]=="y":
                accuracy+=0.45*angle(rl1,rl2,rl3,"s",m.pi)[0]
             elif angle(rl1,rl2,rl3,"a",m.pi)[1]=="y":
                accuracy+=0.45*angle(ll1,ll2,ll3,"s",m.pi)[0]
            
             if angle(la1,la2,la3,"a",m.pi)[1]=="y" :
                accuracy+=0.45*angle(ra1,ra2,ra3,"s",m.pi)[0]
             elif angle(ra1,ra2,ra3,"a",m.pi)[1]=="y":
                accuracy+=0.45*angle(la1,la2,la3,"s",m.pi)[0]
             accuracy+=0.05*touch(ra3,rl3,lm1,rm1)
             accuracy+=0.05*touch(la3,ll3,lm1,rm1)
             return accuracy 
       
       elif yoga=="anantasana":
             if  m.dist(le1,la3)<=m.dist(lm1,rm1)*1.3 :   #  right leg up
                 accuracy+=0.5*touch(ra3,rl3,lm1,rm1)
                 accuracy+=0.25*angle(ll2,ll1,rl1,"r",m.pi/2)[0 ]
                 accuracy+=0.125**angle(ll1,rl1,rl2,"r",m.pi/2)[0]
                 accuracy+=0.125**angle(rl1,rl2,rl3,"r",m.pi/2)[0]
                 
             elif m.dist(re1,ra3)<=m.dist(lm1,rm1)*1.3:
                 accuracy+=0.5*touch(la3,ll3,lm1,rm1)
                 accuracy+=0.25**angle(rl2,rl1,ll1,"r",m.pi/2)[0]                 
                 accuracy+=0.125**angle(ll1,ll2,ll3,"r",m.pi/2)[0]
                 accuracy+=0.125*angle(rl1,ll1,ll2,"r",m.pi/2)[0]
             return accuracy
       
       elif yoga=="anjaneyasana":
             accuracy+=0.25*angle(ra1,ra2,ra3,"s",m.pi)[0] 
             accuracy+=0.25*angle(la1,la2,la3,"s",m.pi)[0]
             if angle(rl1,rl2,rl3,"s",m.pi)[1]=="y":
                 accuracy+=0.125
             if angle(ll1,ll2,ll3,"s",m.pi)[1]=="y":
                 accuracy+=0.125
             accuracy+=0.1*touch(rh3,lh3,lm1,rm1)
             accuracy+=0.1*touch(rh2,lh2,lm1,rm1)
             accuracy+=0.1*touch(rh1,lh1,lm1,rm1)
             return accuracy
        
       elif yoga=="ardha chandrasana":
             accuracy+=0.0625*angle(ra1,ra2,ra3,"s",m.pi)[0] 
             accuracy+=0.0625*angle(la1,la2,la3,"s",m.pi)[0]
             accuracy+=0.0625*angle(la2,la1,ra1,"s",m.pi)[0]
             accuracy+=0.0625*angle(ra2,ra1,la1,"s",m.pi)[0]
             accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
             accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
             if parallelline(ra2,ra1,rl1,rl2,'p')>=parallelline(la2,la1,ll1,ll2,'p'): 
                 accuracy+=0.5*angle(ll2,ll1,rl1,"r",m.pi/2)[1]   
             else:
                 accuracy+=0.5*angle(rl2,rl1,ll1,"r",m.pi/2)[1]
             return accuracy      
       
       elif yoga=="ashtanga namaskara":
           accuracy+=0.125*angle(ra3,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.125*angle(la3,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.25*angle(ra1,rl1,rl2,"r",m.pi/2)[0]
           accuracy+=0.25*angle(la1,ll1,ll2,"r",m.pi/2)[0]
           if angle(la1,la2,la3,"a",m.pi/12)[0]=="y":
               accuracy+=0.125
           if angle(ra1,ra2,ra3,"a",m.pi/12)[0]=="y":
               accuracy+=0.125
           return accuracy    
 
       elif yoga=="baddha konasana":
           accuracy+=0.25*touch(ra3,rl3,lm1,rm1)
           accuracy+=0.25*touch(la3,ll3,lm1,rm1)
           accuracy+=0.25*touch(rl3,ll3,lm1,rm1)
           if angle(ll1,ll2,ll3,"a",m.pi/12)[1]=="y":
               accuracy+=0.125
           if angle(rl1,rl2,rl3,"a",m.pi/12)[1]=="y":
               accuracy+=0.125
           return accuracy    
       
       elif yoga=="bakasana":
           ########  cheeeck moooooooorrrrrrrrrrrreeeeeeeeeee
           accuracy+=0.25*parallelplane(lh1,lh2,lh3,rh1,rh2,rh3)
           accuracy+=0.25*parallelplane(n0,ll3,rl3,rh1,rh2,rh3)
           accuracy+=0.25*parallelline(ra3,ra2,la2,la3,"p")
           accuracy+=0.125*angle(ra2,ra3,rh1,"r",m.pi/2)[0]
           accuracy+=0.125*angle(la2,la3,lh1,"r",m.pi/2)[0]
           return accuracy
       
       elif yoga=="bhujangasana":
           accuracy+=0.15*touch(rl3,ll3,lm1,rm1)
           accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.125*lengthapart(ra3,la3,ra1,la1)
           accuracy+=0.125*parallelline(ra3,ra1,la1,la3,"p")
           if angle(la1,ll1,ll2,"o",m.pi)[1]==angle(ra1,rl1,rl2,"o",m.pi)[1]=="y":
               accuracy+=0.10
           return accuracy    
                       
       elif yoga=="bitilasana":
           accuracy+=0.125*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"s",m.pi)[0]
           accuracy+=0.125*angle(ll1,ll2,ll3,"r",m.pi/2)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"r",m.pi/2)[0]
           accuracy+=0.125*lengthapart(rl3,ll3,rl1,ll1)
           accuracy+=0.125*lengthapart(ra3,la3,ra1,la1)
           accuracy+=0.25*parallelplane(lh1,lh2,lh3,rh1,rh2,rh3)
           return accuracy 
        
       elif yoga=="chaturanga dandasana":
           accuracy+=0.125*angle(la1,la2,la3,"r",m.pi/2)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"r",m.pi/2)[0]
           accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.125*parallelplane(la1,ra1,rl1,ll1,ll3,rl3)
           accuracy+=0.125*parallelline(ra2,ra1,la1,la2,"p")
           accuracy+=0.125*parallelline(ra3,ra2,la2,la3,"p")
           accuracy+=0.125*parallelplane(la1,ra1,ra2,la2,ll3,rl3)
           return accuracy
        
       elif yoga=="dandasana":
           accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.25*parallelplane(la1,ra1,ll1,rf2,rf1,lf1)
           accuracy+=0.125*touch(rl3,ll3,lm1,rm1)
           accuracy+=0.125*touch(rl2,ll2,lm1,rm1)
           accuracy+=0.125*touch(ra3,rl1,lm1,rm1)
           accuracy+=0.125*touch(la3,ll1,lm1,rm1)
           return accuracy
       
       elif yoga=="dhanurasana":
           accuracy+=0.125*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"s",m.pi)[0]
           accuracy+=0.125*touch(ra3,rl3,lm1,rm1)
           accuracy+=0.125*touch(la3,ll3,lm1,rm1)
           accuracy+=0.125*lengthapart(rl3,ll3,rl1,ll1)
           if angle(ll2,ll1,la1,"o",m.pi*3/4)==angle(rl2,rl1,ra1,"o",m.pi*3/4) and angle(rl2,rl1,ra1,"o",m.pi*3/4)[1]=="y":
               accuracy+=0.25
           if angle(ll1,ll2,ll3,"a",m.pi/4)==angle(rl1,rl2,rl3,"a",m.pi/4) and angle(rl1,rl2,rl3,"a",m.pi/4)[1]=="y":
               accuracy+=0.125
           return accuracy
      
       elif yoga=="dwi pada viparita dandasana":
           accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.125*touch(rl3,ll3,lm1,rm1)
           accuracy+=0.125*touch(rl2,ll2,lm1,rm1)
           accuracy+=0.125*touch(ra3,la3,lm1,rm1)
           accuracy+=0.125*angle(la1,la2,la3,"r",m.pi/2)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"r",m.pi/2)[0]
           if angle(ll2,ll1,la1,"o",m.pi*3/4)==angle(rl2,rl1,ra1,"o",m.pi*3/4) and angle(rl2,rl1,ra1,"o",m.pi*3/4)[1]=="y":
               accuracy+=1.25
           return accuracy
       
        
       elif yoga=="halasana":
           accuracy+=0.125*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"s",m.pi)[0]
           accuracy+=0.125*parallelline(ra3,ra1,la1,la3,"p")
           accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.125*touch(rl3,ll3,lm1,rm1)
           accuracy+=0.125*touch(rl2,ll2,lm1,rm1)
           accuracy+=0.025*parallelplane(lf2,rf2,ra1,la1,ra3,la3)
           if angle(ll2,ll1,la1,"a",m.pi/4)==angle(rl2,rl1,ra1,"a",m.pi/4) and angle(rl2,rl1,ra1,"a",m.pi/4)[1]=="y":
               accuracy+=0.1         
           return accuracy
       
       elif yoga=="hanumanasana":
            accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
            accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
            accuracy+=0.125*parallelline(ra3,ra1,la1,la3,"ap")         
            accuracy+=0.125*angle(ll2,ll1,la1,"r",m.pi/2)[0]
            accuracy+=0.125*angle(rl2,rl1,ra1,"r",m.pi/2)[0]
            if angle(la1,la2,la3,"a",m.pi/4)==angle(ra1,ra2,ra3,"a",m.pi/4) and angle(ra1,ra2,ra3,"a",m.pi/4)[1]=="y":
                accuracy+=0.075
            accuracy+=0.1*touch(rh3,lh3,lm1,rm1)
            accuracy+=0.1*touch(rh2,lh2,lm1,rm1)
            accuracy+=0.1*touch(rh1,lh1,lm1,rm1)
            return accuracy
        
       elif yoga=="karnapidasana":
           accuracy+=0.125*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"s",m.pi)[0]
           accuracy+=0.125*parallelline(ra3,ra1,la1,la3,"p")           
           accuracy+=0.125*touch(rl3,ll3,lm1,rm1)
           accuracy+=0.125*touch(rl2,re1,lm1,rm1)
           accuracy+=0.125*touch(ll2,le1,lm1,rm1)
           accuracy+=0.1*parallelplane(lf2,rf2,ra1,la1,ra3,la3)
           if angle(ll2,ll1,la1,"a",m.pi/4)==angle(rl2,rl1,ra1,"a",m.pi/4) and angle(rl2,rl1,ra1,"a",m.pi/4)[1]=="y":
               accuracy+=0.15         
           return accuracy
      
       elif yoga=="natarajasana":
           #### check more
           
           accuracy+=0.25*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.25*angle(ra1,ra2,ra3,"s",m.pi)[0]
           ## either
           if touch(la3,ll3,lm1,rm1):
               accuracy+=0.25*angle(rl1,rl2,rl3,"s",m.pi)[0]
               if angle(ll1,ll2,ll3,"o",m.pi*3/4)[1]=="y":
                   accuracy+=0.25*touch(la3,ll3,lm1,rm1)
           else:
               accuracy+=0.25*angle(ll1,ll2,ll3,"s",m.pi)[0]
               if angle(rl1,rl2,rl3,"o",m.pi*3/4)[1]=="y":
                   accuracy+=0.25*touch(ra3,rl3,lm1,rm1)
           return accuracy
       
        
       elif yoga=="parighasana":
           #     check more  either           
           accuracy+=0.15*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.15*angle(ra1,ra2,ra3,"s",m.pi)[0]
           accuracy+=0.15*touch(rl3,rl2,lm1,rm1)
           accuracy+=0.15*touch(ll2,le1,lm1,rm1)
           accuracy+=0.15*angle(ll1,ll2,ll3,"r",m.pi/2)[0]
           accuracy+=0.15*angle(rl1,rl2,rl3,"s",m.pi)[0]
           if m.dist(ra1,rl1)<m.dist(la1,ll1):
               accuracy+=0.1     
           return accuracy
       
        
       elif yoga=="parivrtta trikonasana":
           accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.125*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"s",m.pi)[0]
           accuracy+=0.25*parallelline(ra3,ra1,la1,la3,"ap")
           if m.dist(la3,rl3)>m.dist(ra3,ll3):
               accuracy+=0.125*touch(ra3,ll3,lm1,rm1)
           else:
               accuracy+=0.125*touch(la3,rl3,lm1,rm1)
           return accuracy
       
        
       elif yoga=="pincha mayurasana":
           accuracy+=0.15*parallelline(rl2,rl1,ll1,ll2,"p")
           accuracy+=0.15*parallelline(ra2,ra1,la1,la2,"p")
           accuracy+=0.15*parallelline(ra3,ra2,la2,la3,"p")
           accuracy+=0.135*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.135*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.135*angle(ra1,ra2,ra3,"r",m.pi/2)[0]
           accuracy+=0.135*angle(la1,la2,la3,"r",m.pi/2)[0]
           if  angle(ra1,rl1,rl2,"s",m.pi)[1]==angle(la1,ll1,ll2,"s",m.pi)[1]=="y" or angle(ra1,rl1,rl2,"o",m.pi)[1]==angle(la1,ll1,ll2,"o",m.pi)[1]=="y":
               accuracy+=0.15
           return accuracy
       
       elif yoga=="ustrasana":
           accuracy+=0.125*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"s",m.pi)[0]
           accuracy+=0.125*touch(ra3,rl3,lm1,rm1)
           accuracy+=0.125*touch(la3,ll3,lm1,rm1)
           accuracy+=0.125*lengthapart(rl3,ll3,rl1,ll1)
           accuracy+=0.125*angle(ll1,ll2,ll3,"r",m.pi/2)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"r",m.pi/2)[0]
           
           if angle(ll2,ll1,la1,"o",m.pi*3/4)==angle(rl2,rl1,ra1,"o",m.pi*3/4) and angle(rl2,rl1,ra1,"o",m.pi*3/4)[1]=="y":
               accuracy+=0.125
           
           return accuracy
               
       elif yoga=="utkatasana":
           accuracy+=0.125*touch(rl3,ll3,lm1,rm1)
           accuracy+=0.125*touch(rl2,ll2,lm1,rm1)
           accuracy+=0.125*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.125*angle(ra1,ra2,ra3,"s",m.pi)[0]          
           accuracy+=0.1*touch(rh3,lh3,lm1,rm1)
           accuracy+=0.1*touch(rh2,lh2,lm1,rm1)
           accuracy+=0.1*touch(rh1,lh1,lm1,rm1)
           if angle(ll2,ll1,la1,"o",m.pi*3/4)==angle(rl2,rl1,ra1,"o",m.pi*3/4) and angle(rl2,rl1,ra1,"o",m.pi*3/4)[1]=="y":
               accuracy+=0.1
           if angle(ll1,ll2,ll3,"a",m.pi/4)==angle(rl1,rl2,rl3,"a",m.pi/4) and angle(rl1,rl2,rl3,"a",m.pi/4)[1]=="y":
               accuracy+=0.1
           
           return accuracy
       
        
       elif yoga=="utthita hasta padangustasana":
           accuracy+=0.125*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.125*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.25*parallelline(ra3,ra1,la1,la3,"ap")
           if m.dist(la3,ll3)>m.dist(ra3,rl3):
               accuracy+=0.1*touch(ra3,rl3,lm1,rm1)
               accuracy+=0.1*touch(la3,ll1,lm1,rm1)
               accuracy+=0.1*angle(la1,la2,la3,"a",m.pi)[0]
               accuracy+=0.1*angle(ra1,ra2,ra3,"s",m.pi)[0]
               accuracy+=0.1*angle(la1,ll1,ll2,"s",m.pi)[0]
               
           else:
               accuracy+=0.1*touch(la3,ll3,lm1,rm1)
               accuracy+=0.1*touch(ra3,rl1,lm1,rm1)
               accuracy+=0.1*angle(ra1,ra2,ra3,"a",m.pi)[0]
               accuracy+=0.1*angle(la1,la2,la3,"s",m.pi)[0]
               accuracy+=0.1*angle(ra1,rl1,rl2,"s",m.pi)[0]
           return accuracy
       
       elif yoga=="virabhadrasana II":
           accuracy+=0.2*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.2*angle(ra1,ra2,ra3,"s",m.pi)[0]
           accuracy+=0.2*parallelline(ra3,ra1,la1,la3,"ap")
           accuracy+=0.2*angle(ll1,ll2,ll3,"r",m.pi/2)[0]
           accuracy+=0.2*angle(rl1,rl2,rl3,"s",m.pi)[0]
           return accuracy
               
       elif yoga=="virabhadrasana III":
           accuracy+=0.1*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.1*angle(ra1,ra2,ra3,"s",m.pi)[0]          
           accuracy+=0.1*touch(rh3,lh3,lm1,rm1)
           accuracy+=0.1*touch(rh2,lh2,lm1,rm1)
           accuracy+=0.1*touch(rh1,lh1,lm1,rm1)
           accuracy+=0.1*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.1*angle(rl1,rl2,rl3,"s",m.pi)[0]
           accuracy+=0.2*angle(rl1,ll1,ll2,"r",m.pi/2)[0]
           if angle(ra1,la1,la2,"o",m.pi/2)==angle(la1,ra1,ra2,"o",m.pi/2) and angle(ra1,la1,la2,"o",m.pi/2)[1]=="y":
                  accuracy+=0.1
           return accuracy
       
        
       elif yoga=="vriksasana":
           accuracy+=0.1*angle(la1,la2,la3,"s",m.pi)[0]
           accuracy+=0.1*angle(ra1,ra2,ra3,"s",m.pi)[0]          
           accuracy+=0.1*touch(rh3,lh3,lm1,rm1)
           accuracy+=0.1*touch(rh2,lh2,lm1,rm1)
           accuracy+=0.1*touch(rh1,lh1,lm1,rm1)
           accuracy+=0.1*angle(ll1,ll2,ll3,"s",m.pi)[0]
           accuracy+=0.1*angle(rl1,rl2,rl3,"s",m.pi)[0]
           if angle(rl1,ll1,ll2,"a",m.pi/2)[1]=="y":
                  accuracy+=0.2*touch(rf1,ll2,lm1,rm1)
           if angle(ra1,la1,la2,"o",m.pi/2)==angle(la1,ra1,ra2,"o",m.pi/2) and angle(ra1,la1,la2,"o",m.pi/2)[1]=="y":
                  accuracy+=0.1
           return accuracy
        

            
         
            
         
            
       
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

yoga="bitilasana" 
# For webcam input:
vid = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('{}.mp4'.format(yoga), fourcc, 60.0, (640, 480))
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
  ret, image = vid.read()
  
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
        if acc>=0.8:
             print("!!!!!!!!!!!!!!!!!!!!!!!!check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        cv2.putText(image,str(int(acc*100))+"%",( int(image.shape[0]/2) ,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    cv2.imshow('MediaPipe Pose', image)
    out.write(image)
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
        m1=load_model("Modals/Final(1).h5")
        #print(m1.summary())
        pred=m1(img)
        pred=np.array(pred)
        pred=pred.argmax()
        pred=int(pred)
        yoga1={0: 'adho mukha svanasana', 1: 'adho mukha vriksasana', 2: 'akarna dhanurasana', 3: 'anantasana', 4: 'anjaneyasana', 5: 'ardha chandrasana', 6: 'ashtanga namaskara', 7: 'baddha konasana', 8: 'bakasana', 9: 'bhujangasana', 10: 'bitilasana', 11: 'chaturanga dandasana', 12: 'dandasana', 13: 'dhanurasana', 14: 'dwi pada viparita dandasana', 15: 'halasana', 16: 'hanumanasana', 17: 'karnapidasana', 18: 'natarajasana', 19: 'parighasana', 20: 'parivrtta trikonasana', 21: 'pincha mayurasana', 22: 'ustrasana', 23: 'utkatasana', 24: 'utthita hasta padangustasana', 25: 'virabhadrasana ii', 26: 'virabhadrasana iii', 27: 'vriksasana'}
        yoga=yoga1[pred]
        print(yoga1[pred])
        
out.release()       
vid.release()
cv2.destroyAllWindows() 
print("done")