import cv2
from ultralytics import YOLO
import cvzone
import math

#cap=cv2.VideoCapture(0) #using one webcam
cap=cv2.VideoCapture('../videos/MUNICH_dashcam.mp4')# for videos
cap.set(3,1280) #width
cap.set(4,720) #height
#model
model = YOLO('../yolo_weights/yolov8n.pt')

#detecting the class from koko dataset
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup"
              ]

while True:
    success, img=cap.read()
    results=model(img,stream=True)# stream = True will use generators and will be efficient
    #creating bounding boxex
    for r in results:
        boxes=r.boxes #bounding box for each result
        for box in boxes:#loop through boxes

            # making the bounding boxex
            x1,y1,x2,y2 = box.xyxy[0]#finding the (x,y) for each box
            x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)#convert the co-ordinates to int to use them

            # for open cv
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # image,(points...),(color),thickness
            #print(x1, y1, x2, y2)

            #for cvzone
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img, (x1,y1,w,h))
            print(x1,y1,w,h)
            #confidence values
            conf= math.ceil((box.conf[0]*100))/100 #ceil is to round the conf values
            print(conf)
            #display the confidence and class name; displaying it on a rectangle (text format)
            #class name
            cls=int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1) #x1 and y1 is to display the conf in the starting point of the bounding box, and max is to prevent the cong values to go out of the screen

    cv2.imshow('video',img)
    #if 'q' is pressed pop up is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()