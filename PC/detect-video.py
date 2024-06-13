import cv2
import time

import torch
from ultralytics import YOLO
from torchinfo import summary

# use gpu if is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

pt_file = 'yolov8n.pt'
model = YOLO(pt_file)
model.to(device)
# summary(model)


cap = cv2.VideoCapture('./data/veh_640x480.mp4')
# cap = cv2.VideoCapture('./data/veh2.mp4')

result_time = []
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            start = time.time()
            res = model.predict(frame, conf=0.5, verbose=True)
            end = time.time()
            t = end-start
            result_time.append(t)
            if(t>0):
                fps = 1/(end-start)

            res_plotted = res[0].plot()
            cv2.putText(res_plotted, "FPS:"+str(fps), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            cv2.imshow("result", res_plotted)

            key = cv2.waitKey(33)       # 
            if key == ord('q'):         # 
                break
        else:
            break
finally:
    print(result_time)

cap.release()
cv2.destroyAllWindows()
