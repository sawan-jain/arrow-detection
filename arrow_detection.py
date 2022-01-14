import cv2 as cv
import numpy as np

def detectRed(img,hsv,font) :
    #lower_blue=np.array([0,130,184]) # B=144
    #upper_blue=np.array([179,255,255])
    lower_blue=np.array([0,200,140]) # B=144 G=220
    upper_blue=np.array([185,255,255]) # R=175

    mask=cv.inRange(hsv,lower_blue,upper_blue)
    bluents,heirarchy=cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if(len(bluents)>0) :
        blue_area=max(bluents,key=cv.contourArea)
        approx=cv.approxPolyDP(blue_area,0.01*cv.arcLength(blue_area,True),True)
        if(len(approx)==7) :
            (x,y,w,h)=cv.boundingRect(blue_area)
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,2),2)
            cv.putText(img,"arrow",(x,y),font,1,(220,0,0),2,cv.LINE_AA)

    cv.imshow("mask1",mask)

    return img

def main() :
    img=cv.imread('photos/test_img_2/test5.jpg')
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    font=cv.FONT_HERSHEY_COMPLEX
    cv.imshow("hsv",hsv)
    cv.imshow("final",detectRed(img,hsv,font))
    
    cv.waitKey(0)
    if cv.waitKey(1) and 0xFF==ord('q') :
        cv.destroyAllWindows()   


if __name__=="__main__" :
    main()






'''
    cap=cv.VideoCapture(1)
    font=cv.FONT_HERSHEY_COMPLEX
    while True:
        _,frame=cap.read()
        hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        if cv.waitKey(20) and 0xFF==ord('d') :
            #cap.release()
            cv.destroyAllWindows()
            break
        cv.imshow("img1",detectRed(frame,hsv,font))

       '''
    #cv.destroyAllWindows()