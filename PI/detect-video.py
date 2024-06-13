

import cv2

import torch
from ultralytics import YOLO
import time
from ultralytics.utils.plotting import Annotator, colors, save_one_box


pt_file = 'yolov8n.pt'
model = YOLO(pt_file)


cap = cv2.VideoCapture('./data/veh_640x480.mp4')


while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        start = time.time()
        res = model(frame, conf=0.5, verbose=False)
        end = time.time()
        fps = 1/(end-start)

        res_plotted = res[0].plot()

        cv2.putText(res_plotted, "FPS:"+str(fps), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow("result", res_plotted)


        key = cv2.waitKey(25)
        if key == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()


