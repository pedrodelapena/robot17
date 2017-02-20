#!/usr/bin/env python
__author__      = "Matheus Dib, Fabio de Miranda"


import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt
import time

import itertools

# If you want to open a video, just change this path
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


persistence= []
keepOn= True
font = cv2.FONT_HERSHEY_SIMPLEX
while(keepOn):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    # A gaussian blur to get rid of the noise in the image
    # Detect the edges present in the image
    # Obtains a version of the edges image where we can draw in color
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    bordas = auto_canny(blur)
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    


    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles=cv2.HoughCircles(bordas,cv.CV_HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=2,maxRadius=256)
    
    if circles == None:
        circles=[[]]
    circles = np.uint16(np.around(circles))[0]
    
    #segunda filtragem: só separar os três círculos relevantes
    if len(circles) >=3:
             
        ways= list( itertools.combinations(circles, 3) )
        
        i=0
        while(i!=len(ways)):
            
            #conferir as ways com círculos de tamanho +=10%
            #pode ser outro valor além de 10%
            
            ref= float(1)*ways[i][0][2]*ways[i][1][2]*ways[i][2][2]
            
            check1= ((ref-ways[i][0][2]**3)/ref)**2
            check2= ((ref-ways[i][1][2]**3)/ref)**2
            check3= ((ref-ways[i][2][2]**3)/ref)**2
                    
            tolerance= 0.3
            
            if check1 > tolerance or check2 > tolerance or check2 > tolerance:
                ways.pop(i)
                i-=1
                
            i+=1
        
        '''
        i=0
        while(i!=len(ways)):
            
            #conferir as ways com colinearidade +-10%
            #pode ser outro valor além de 10%
            
            #distancia x
            x= max(circles[0][0], circles[1][0]) - min(circles[0][0], circles[1][0])
            x= float(x)+1
            #distancia y
            y= max(circles[0][1], circles[1][1]) - min(circles[0][1], circles[1][1])
            y= float(y)+1
                    
            check1= x/y
            
            #distancia x
            x= max(circles[0][0], circles[2][0]) - min(circles[0][0], circles[1][0]) 
            x= float(x)+1
            #distancia y
            y= max(circles[0][1], circles[2][1]) - min(circles[0][1], circles[1][1])
            y= float(y)+1
            
            check2= x/y
            
            #distancia x
            x= max(circles[1][0], circles[2][0]) - min(circles[0][0], circles[1][0])
            x= float(x)+1
            #distancia y
            y= max(circles[1][1], circles[2][1]) - min(circles[0][1], circles[1][1])
            y= float(y)+1
            
            check3= x/y
                         
                         
            ref= check1*check2*check3
            
            check1= ((ref-check1**3)/ref)**2
            check2= ((ref-check2**3)/ref)**2
            check3= ((ref-check3**3)/ref)**2
                    
            print(check1)
            
            
                    
            tolerance= 10
            
            if check1 > tolerance or check2 > tolerance or check2 > tolerance:
                ways.pop(i)
                i-=1
                
            i+=1
        '''
        
        if len(ways)!=0:
            circles= ways[0]
        else:
            circles=[[]]
            circles = np.uint16(np.around(circles))[0]
    else:
        circles=[[]]
        circles = np.uint16(np.around(circles))[0]
    
    acc=0
    for circle in circles:
        #vamos medir as distâncias aqui
        #96px = 20cm
        acc+=1920.0/circle[2]
    num= str(round(acc/3, 2))
    cv2.putText(bordas_color,num+" cm",(0,50), font, 2,(255,255,255),2,cv2.CV_AA)
    
    #vamos medir aqui o ângulo
    if len(circles) >= 2:
        #distancia x
        x= max(circles[0][0], circles[1][0]) - min(circles[0][0], circles[1][0]) 
        #distancia y
        y= max(circles[0][1], circles[1][1]) - min(circles[0][1], circles[1][1])
        
        #distancia absoluta
        hipo= (x**2+y**2)**(1.0/2)
        if ((float(y)/hipo)>= (2.0**(1.0/2.0)/2)):
            cv2.putText(bordas_color,"Vertical",(0,460), font, 2,(255,0,0),2,cv2.CV_AA)
        else:
            cv2.putText(bordas_color,"Horizontal",(0,460), font, 2,(0,255,0),2,cv2.CV_AA)
        #print("sen "+str(float(y)/hipo))
        
    for i in circles:
        # draw the outer circle
        # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
        cv2.circle(bordas_color,(i[0],i[1]),i[2],(255,128,0),2)
        # draw the center of the circle
        cv2.circle(bordas_color,(i[0],i[1]),2,(0,128,255),3)

    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    #cv2.line(bordas_color,(0,0),(511,511),(255,0,0),5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    #cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(bordas_color,'Ninjutsu ;)',(0,50), font, 2,(255,255,255),2,cv2.CV_AA)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('Detector de circulos',bordas_color)    
    keepOn= not(cv2.waitKey(1) & 0xFF == ord('q'))
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()