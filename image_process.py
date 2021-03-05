import cv2
import numpy as np
import statistics 

e1=cv2.getTickCount()
img=cv2.imread('One.jpg')


hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow('img',img)
cv2.namedWindow('img')
arr=[]

def draw1(event,x,y,flags,param):
    global ix,iy

    if event==cv2.EVENT_LBUTTONDOWN:
        arr.append(list(hsv[y,x]))
cv2.setMouseCallback('img',draw1)
while (1):
    k=cv2.waitKey(1)
    if k==ord('q'):
        cv2.destroyAllWindows()
        break

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

lower_brown=np.array([5,50,50])
upper_brown=np.array([100,255,255])
mask=cv2.inRange(hsv,lower_brown,upper_brown)
res=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow('img',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

cv2.imshow('gray',gray)
gray=cv2.medianBlur(gray,5) #to remove noise/ random black spots
gray1=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('gray1',gray1)
gray2=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('gray2',gray2)


dst1=cv2.bilateralFilter(img,9,75,75)
cv2.imshow('bilateral filtering', dst1) #removes noise but keeps edges sharp
# to get canny edges : 1) Filter to remove noise, 2) calculate image gradients
# 3) Image gradient maximum at boundary, 4) Use limits maxval and minval
# 5) edges have gradient more than maxval, 6) ALL done by canny edge function
edges=cv2.Canny(res,100,200)
cv2.imshow('edges',edges)

#contours
contours,x=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask,contours,-1,(255,255,255),2)
cv2.imshow('con',mask)

#Harris Corner detection
img1=img.copy()
a=cv2.cornerHarris(gray,2,3,0.04)
img1[a>0.01*a.max()]=[0,255,255]
cv2.imshow('a',img1)
img1=img.copy()

#ShiTomasi Corners using goodFeaturesToTrack()

corners=cv2.goodFeaturesToTrack(gray,100,0.01,10) #second argument is number of corners to track
corners=np.int0(corners)
for i in corners:
    x,y=i.ravel()
    cv2.circle(img1,(x,y),3,(255,0,0),-1)
cv2.imshow('b',img1)

#FAST object
img1=img.copy()
fast=cv2.FastFeatureDetector_create()
kp=fast.detect(gray,None)
img1=cv2.drawKeypoints(img1,kp,None,(255,0,0))
cv2.imshow('c',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Numpy numbering example + modification on image
i=img
head=i[:200,50:190]
cv2.imshow('i',head)
cv2.line(img,(0,0),(0,200),(0,255,0),10)
cv2.circle(img,(0,0),100,(255,0,0),-1)
f=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'MEE',(0,300),f,4,(255,255,0),2,cv2.LINE_AA)


cv2.imshow('image',img)


