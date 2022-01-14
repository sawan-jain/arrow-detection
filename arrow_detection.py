import cv2 as cv
import numpy as np

def detectRed(img,hsv,font) :
    
    lower_blue=np.array([0,200,140]) # B=144 G=220
    upper_blue=np.array([185,255,255]) # R=175

    mask=cv.inRange(hsv,lower_blue,upper_blue)        # to filter out only the required colour from the image/frame
    bluents,heirarchy=cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if(len(bluents)>0) :
        blue_area=max(bluents,key=cv.contourArea)
        approx=cv.approxPolyDP(blue_area,0.01*cv.arcLength(blue_area,True),True)
        if(len(approx)==7) :
            (x,y,w,h)=cv.boundingRect(blue_area)
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,2),2)
            cv.putText(img,"arrow",(x,y),font,1,(220,0,0),2,cv.LINE_AA)

    cv.imshow("mask1",mask)                          # to check whether it is sensing or not

    return img

def main() :
    img=cv.imread('photos/test_img_2/test5.jpg')     # image path 
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)            # converted from BGR to HSV
    font=cv.FONT_HERSHEY_COMPLEX                     # font for text
    cv.imshow("hsv",hsv)                             # showed hsv for analysis
    cv.imshow("final",detectRed(img,hsv,font))
    
    cv.waitKey(0)
    if cv.waitKey(1) and 0xFF==ord('q') :            # to stop the function when a key id pressed
        cv.destroyAllWindows()   


if __name__=="__main__" :
    main()

'''   FOR VIDEO CAPTURE

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

      
    #cv.destroyAllWindows()
    
 '''
