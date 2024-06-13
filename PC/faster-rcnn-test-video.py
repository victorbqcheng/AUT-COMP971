import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt

import torchvision

from PIL import Image
import cv2

import time

names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT, pretrained=True)
model.eval()

threshold = 0.5

def predictImg(img):
    np_img = np.array(img)
    tensor_img = torch.from_numpy(np_img/255).permute(2, 0, 1).type(torch.float32)
    
    start = time.time()
    pred = model([tensor_img])
    end = time.time()
    t1 = end-start

    boxes = pred[0]['boxes']
    labels = pred[0]['labels']
    scores = pred[0]['scores']

    pred_index = scores > threshold
    boxes = boxes[pred_index]
    labels = labels[pred_index]

    car_index = labels == 3
    boxes = boxes[car_index]
    labels = labels[car_index]

    labels = [names.get(str(idx.item())) for idx in labels]

    img = torch.from_numpy(np_img).permute(2, 0, 1)

    result = torchvision.utils.draw_bounding_boxes(
                            img,
                            boxes,
                            labels)
    return result.permute(1,2,0).numpy(), 1/t1


cap = cv2.VideoCapture('./data/veh_640x480.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        result, fps = predictImg(frame)
        cv2.putText(result, "FPS:"+str(fps), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow('result', result)
        key = cv2.waitKey(25)       # 
        if key == ord('q'):         # 
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


