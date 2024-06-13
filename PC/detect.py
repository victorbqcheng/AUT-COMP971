import cv2
import time

from ultralytics import YOLO
import onnx

import matplotlib.pyplot as plt
import torch
import pandas as pd
from tqdm import tqdm
from torchinfo import summary

import cProfile

profiler = cProfile.Profile()

# use gpu if is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

pt_file = 'yolov8n.pt'
model = YOLO(pt_file)
model.to(device)
summary(model)
# model.export(format='onnx', opset=10)
# model.export(format='ncnn')

# img_path = "./data/test/yellow-audi899johme.png"
# img_path = "./data/test/car-and-bus.jpg"
# img_path = "./data/test/car1.jpg"
# img_path = "./data/test/car2.jpg"
# img_path = "./data/test/car-1920x1080.jpg"
# img_path = "./data/test/car-1280x720.jpg"
# img_path = "./data/test/car-720x480.jpg"


class Obj:
    def __init__(self, img_path, csvfile) -> None:
        self.img_path = img_path
        self.csvfile = csvfile
        pass

test_cases = [
    Obj('./data/test/car-1280x720.jpg', 'yolov8n-pc-1280x720'),
    Obj('./data/test/car-1920x1080.jpg', 'yolov8n-pc-1920x1080'),
    Obj('./data/test/car-720x480.jpg', 'yolov8n-pc-720x480'),
    Obj('./data/test/car-640x426.jpg', 'yolov8n-pc-640x426'),
    Obj('./data/test/car-360x240.jpg', 'yolov8n-pc-360x240'),
    ]
    

def test(img_path, csvfile):
    img = cv2.imread(img_path)

    result_time = []
    df = pd.DataFrame([])

    for i in tqdm(range(1)):
        # print(i)
        start = time.time()
        # profiler.enable()
        results = model(img, verbose=True, conf=0.5, save=True)
        # profiler.disable()
        # results = model.predict(source=img, conf=0.5, verbose=False)
        end = time.time()
        result_time.append(end-start)

    # profiler.dump_stats('cProfile.prof')

    print('inference time:', result_time)
    
    df[csvfile] = result_time
    # df.to_csv(csvfile+'.csv', index=False)
    return


for obj in test_cases:
    img_path, csvfile = obj.img_path, obj.csvfile
    print('img_path:', img_path)
    print('csvfile:', csvfile)
    test(img_path, csvfile)
print('done')


