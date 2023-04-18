import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import supervision as sv
from sort import *

end_time = 0
width_img = 1280
height_img = 720
model = YOLO(r'D:\Python\subline\project\trial\Yolo Weight\yolov8l.pt')
cap = cv2.VideoCapture(r"C:\Users\plado\Downloads\3.mp4")  # For Video

mask = cv2.imread(r"C:\Users\plado\Downloads\Image Processing\mask.png")
mask = cv2.resize(src=mask, dsize=(width_img, height_img))
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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


# Tracking
tracker = Sort(max_age=36, min_hits=3, iou_threshold=0.48)

limits = [800, 297, 1200, 297]
totalCount = []


while True:
    success, img = cap.read()
    
    img = cv2.resize(src=img, dsize=(width_img, height_img))
    imgRegion = cv2.bitwise_and(img, mask)
    detections = np.empty((0, 5))

    results = model(imgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" :
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        
        # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                        #    scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
    
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)


        # cv2.polylines(img, [rectangle], False, (0,255,0), thickness=3)
    cv2.imshow("iamge",img)
    
   


    if cv2.waitKey(1) == 27:

        break 
cv2.destroyAllWindows()
cap.release()
	