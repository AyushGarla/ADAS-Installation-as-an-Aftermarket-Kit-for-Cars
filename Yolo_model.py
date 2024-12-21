import cv2
from ultralytics import YOLO
model = YOLO('../yolo_weights/yolov8l.pt') #large version, slower and accurate
#model = YOLO('../yolo_weights/yolov8n.pt') #nano version, faster but not accurate
result=model('Images/download.jpeg',show=True)
cv2.waitKey(0)
