import cv2
from ultralytics import YOLO
import cvzone
import math

#cap = cv2.VideoCapture(0)  # Using webcam
cap = cv2.VideoCapture('../videos/dashcam.mp4') # For videos
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# Model
model = YOLO('../yolo_weights/yolov8n.pt')

# Detecting the class from coco dataset
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
]

while True:
    success, img = cap.read()
    results = model(img, stream=True)  # stream = True will use generators and will be efficient
    # Creating bounding boxes
    for r in results:
        boxes = r.boxes  # Bounding box for each result
        for box in boxes:  # Loop through boxes
            # Making the bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]  # Finding the (x,y) for each box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert the coordinates to int to use them

            # For cvzone
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            print(x1, y1, w, h)

            # Confidence values
            conf = math.ceil((box.conf[0] * 100)) / 100  # Ceil is to round the conf values
            print(conf)

            # Display the confidence and class name; displaying it on a rectangle (text format)
            # Class name
            cls = int(box.cls[0])
            print(f'Detected class: {cls}')

            # Check if cls is within the range of classNames
            if 0 <= cls < len(classNames):
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=3, thickness=3)
            else:
                print(f'Class index {cls} is out of range')

    cv2.imshow('video', img)
    # If 'q' is pressed, the pop-up is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
