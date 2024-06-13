
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

import torch
import time
import pandas as pd
from tqdm import tqdm



pt_file = 'yolov8n.pt'
model = YOLO(pt_file)
# model.export(format='onnx')


img_path = "./data/test/car-and-bus.jpg"
# img_path = "./data/test/car1.jpg"
# img_path = "./data/test/car2.jpg"

class Obj:
    def __init__(self, img_path, csvfile) -> None:
        self.img_path = img_path
        self.csvfile = csvfile
        pass

test_cases = [
    Obj('./data/test/car-360x240.jpg', 'yolov8n-360x240'),
    Obj('./data/test/car-640x426.jpg', 'yolov8n-640x426'),
    Obj('./data/test/car-720x480.jpg', 'yolov8n-720x480'),
    Obj('./data/test/car-1280x720.jpg', 'yolov8n-1280x720'),
    Obj('./data/test/car-1920x1080.jpg', 'yolov8n-1920x1080'),
    ]

def test(img_path, csvfile):
    img = cv2.imread(img_path)
    result_time = []
    
    for i in tqdm(range(500)):
        # print(i)
        start = time.time()
        results = model(img, verbose=False)
        end = time.time()
        result_time.append(end-start)


    # print('predict time:', result_time)

    df = pd.DataFrame([])

    df[csvfile] = result_time
    df.to_csv(csvfile+'.csv', index=False)

    return

for obj in test_cases:
    img_path, csvfile = obj.img_path, obj.csvfile
    print('img_path:', img_path)
    print('csvfile:', csvfile)
    test(img_path, csvfile)
print('done')
