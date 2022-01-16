import cv2 as cv
import numpy as np

def findAngle(img,hsv,font) :
    # boolean variable to check if a valid arrow is present or not
    check=True

    # numpy array which contains the lower and the maximum value of the hsv colour of red , ie, in the yellow region
    # values are entered int BGR format and NOT RGB
    lower_yellow=np.array([0,200,140]) 
    upper_yellow=np.array([185,255,255]) 

    # cv.inRange() will filter out the required colour from the binary image
    mask=cv.inRange(hsv,lower_yellow,upper_yellow)

    # to find contours from the obtained filtered image mask
    yellowents,heirarchy=cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if(len(yellowents)>0) :
        # found contour with max area 
        yellow_area=max(yellowents,key=cv.contourArea)

       # find maximum contour area
        area=cv.contourArea(yellow_area)
    
        # to find the contour with specified precision
        approx=cv.approxPolyDP(yellow_area,0.01*cv.arcLength(yellow_area,True),True)

        # to find arrow with 7 edges
        # if just angle is required without any restrictions then remove the if statement
        if(len(approx)==7) :

            # arrow with given conditions is present so change value of check from True to False
            check=False

            # cv.minAreaRect() is used to find the area of the minimum rect which encloses the contour
            # cv.minAreaarrow returns values in the order:(center(x, y), (width, height), angle of rotation wrt to x-axis)
        
            rect = cv.minAreaRect(yellow_area[0])
            center = (int(rect[0][0]),int(rect[0][1])) 
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = int(rect[2])

            # angle calculation wrt to y-axis
            if width < height:
                angle = 90-angle
            else:
                angle = -angle
            
            # to print angle
            label = "  Rotation Angle: " + str(90-angle) + " degrees"

            # cv.boundingRect() is used to find the horizontal rectangle whch encloses the figure
            # this function returns the centre(x,y),width and height of the rectangle
            (x,y,w,h)=cv.boundingRect(yellow_area)
            textbox = cv.rectangle(img, (x-15, y-25), (center[0] + 295, center[1] + 10), (255,255,255), -1)
            cv.putText(img, label, (x,y), font, 0.7, (0,0,0), 1, cv.LINE_AA)

    if check==False :
        cv.imshow("mask",mask)
        cv.imshow("final",img)
        
    else :
        print("INCORRECT ARROW")


def main() :
    
    # image reading 
    img=cv.imread('/home/user/Desktop/sawan code/arrow detection/test_img_2/test4.png')   
    
    #converting image from BGR to HSV for analysis and detection
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    cv.imshow("hsv",hsv)
    
    font=cv.FONT_HERSHEY_COMPLEX
    findAngle(img,hsv,font)
    cv.waitKey(0)

    # to stop the function when any key on keyboard is pressed
    if cv.waitKey(1) and 0xFF==ord('q') :
        cv.destroyAllWindows()  
    
    '''  FOR VIDEO CAPTURE
    cap=cv.VideoCapture(0)
    font=cv.FONT_HERSHEY_COMPLEX
    while True:
        _,frame=cap.read()
        cv.imshow("frame",frame)
        hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        cv.imshow("hsv",hsv)
        detectRed(frame,hsv,font)
        #cv.imshow("img1",detectRed(frame,hsv,font))
        if cv.waitKey(20) & 0xFF==ord('d') :
            break

    cv.destroyAllWindows()

    '''

if __name__=="__main__" :
    main()
