

import cv2

import torch
from ultralytics import YOLO
from deepsparse import Pipeline
import time

# model_path = "./yolov8n.onnx"
# yolo_pipeline = Pipeline.create(
#     task="yolov8",
#     model_path=model_path,
#     batch_size=1
# )

task = "yolov8"
stub = "./yolov8n.onnx"
yolo_pipeline = Pipeline.create(task, model_path=stub, batch_size=1)


cap = cv2.VideoCapture('./data/veh_640x480.mp4')

count = 0
carid = 2

while cap.isOpened():
    ret, frame = cap.read()
    count +=1
    
    if ret:
            
        start = time.time()
        res = yolo_pipeline(images=[frame])  # YOLOOutput
        end = time.time()
        fps = 1/(end-start)
        boxes = res.boxes[0]
        labels = res.labels[0]
        scores = res.scores[0]
        len(boxes)
        for i in range(len(boxes)):
            if int(float(labels[i]) ) == carid:
                box = boxes[i]
                top_left = (int(box[0]), int(box[1]))
                bottom_right = (int(box[2]), int(box[3]))
                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
                cv2.putText(frame, "car {:.2f}".format(scores[i]), (int(box[0]), int(box[1])-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, "FPS:"+str(fps), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow("result", frame)
        
        key = cv2.waitKey(25) 
        if key == ord('q'):         
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()



