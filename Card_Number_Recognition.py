##############################
###   Bank card number identification
##############################

import cv2
import numpy as np

card_GRAY=cv2.imread('bank_card41.jpg',0)
card_GRAY4=cv2.resize(card_GRAY,(4*card_GRAY.shape[1],4*card_GRAY.shape[0]))
#cv2.imshow('card_GRAY4',card_GRAY4)

adaptiveThresh = cv2.adaptiveThreshold(card_GRAY4, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 13, 3)
#cv2.imshow('adaptiveThresh',adaptiveThresh)

contours, hierarchy = cv2.findContours(adaptiveThresh,cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if cv2.contourArea(contours[i])<160:
        adaptiveThresh = cv2.drawContours(adaptiveThresh, contours, i, (0,0,0), -1)
#cv2.imshow('adaptiveThresh2',adaptiveThresh)

kernel = np.ones((15,15),dtype=np.uint8)
blackhat = cv2.morphologyEx(adaptiveThresh, cv2.MORPH_BLACKHAT, kernel)
#cv2.imshow('blackhat',blackhat)

kernel = np.ones((3,3),dtype=np.uint8)
opening = cv2.morphologyEx(blackhat, cv2.MORPH_OPEN, kernel)
#cv2.imshow('opening',opening)

contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])
    aspect_ratio = float(w)/h
    Area = w * h
    if Area<1800 or Area>6000:
        opening = cv2.drawContours(opening, contours, i, (0,0,0), -1)
    else:
        if aspect_ratio>0.7 or aspect_ratio<0.5:
            opening = cv2.drawContours(opening, contours, i, (0,0,0), -1)
cv2.imshow('opening2',opening)

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(opening,kernel,iterations = 1)
cv2.imshow('dilation',dilation)

numTemplate=cv2.imread('bankCardNumTemplate.jpg')
numTemplate_GRAY=cv2.cvtColor(numTemplate,cv2.COLOR_BGR2GRAY)
ret,numTemplate_GRAY = cv2.threshold(numTemplate_GRAY,200,255,cv2.THRESH_BINARY)
cv2.imshow('numTemplate_GRAY',numTemplate_GRAY)

def sequence_contours(image, width, height):
    contours, hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)
    RectBoxes0 = np.ones((n,4),dtype=int)
    for i in range(n):
        RectBoxes0[i] = cv2.boundingRect(contours[i])
    
    RectBoxes = np.ones((n,4),dtype=int)
    for i in range(n):
        sequence = 0
        for j in range(n):
            if RectBoxes0[i][0]>RectBoxes0[j][0]:
                sequence = sequence + 1
        RectBoxes[sequence] = RectBoxes0[i]
        
    ImgBoxes = [[] for i in range(n)]
    for i in range(n):
        x,y,w,h = RectBoxes[i]
        ROI = image[y:y+h,x:x+w]
        ROI = cv2.resize(ROI, (width, height))
        thresh_val, ROI = cv2.threshold(ROI, 200, 255, cv2.THRESH_BINARY)
        ImgBoxes[i] = ROI
        
    return RectBoxes, ImgBoxes
    
RectBoxes_Temp, ImgBoxes_Temp = sequence_contours(numTemplate_GRAY, 50, 80)
print(RectBoxes_Temp)
cv2.imshow('ImgBoxes_Temp[1]',ImgBoxes_Temp[1])
RectBoxes, ImgBoxes = sequence_contours(dilation, 50, 80)

result = []
for i in range(len(ImgBoxes)):
    score = np.zeros(len(ImgBoxes_Temp),dtype=int)
    for j in range(len(ImgBoxes_Temp)):
        score[j] = cv2.matchTemplate(ImgBoxes[i], ImgBoxes_Temp[j], cv2.TM_SQDIFF)
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(score)
    result.append(min_indx[1])
print(result)

cv2.waitKey(0)
cv2.destroyAllWindows()




