import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary


import torchvision

from PIL import Image

import time

import pandas as pd


names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT, pretrained=True)

x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
torch.onnx.export(model,x, "faster-rcnn.onnx", opset_version=11)
model.eval()
info = summary(model)



img_path = './data/test/car-and-bus.jpg'
img_path = './data/test/car1.jpg'
img_path = './data/test/car2.jpg'
img_path = "./data/test/car-1920x1080.jpg"
img_path = "./data/test/car-1280x720.jpg"
# img_path = "./data/test/car-720x480.jpg"
# img_path = "./data/test/car-640x426.jpg"
# img_path = "./data/test/car-360x240.jpg"

class Obj:
    def __init__(self, img_path, csvfile) -> None:
        self.img_path = img_path
        self.csvfile = csvfile
        pass

test_cases = [
    Obj('./data/test/car-1280x720.jpg', 'faster-rcnn-pc-1280x720'),
    Obj('./data/test/car-1920x1080.jpg', 'faster-rcnn-pc-1920x1080'),
    Obj('./data/test/car-720x480.jpg', 'faster-rcnn-pc-720x480'),
    Obj('./data/test/car-640x426.jpg', 'faster-rcnn-pc-640x426'),
    Obj('./data/test/car-360x240.jpg', 'faster-rcnn-pc-360x240'),
    ]

def test(img_path, csvfile):
    pil_img = Image.open(img_path)
    np_img = np.array(pil_img)
    # plt.imshow(np_img)
    # plt.show()

    tensor_img = torch.from_numpy(np_img/255).permute(2, 0, 1).type(torch.float32)
    device = 'cuda:0'
    tensor_img = tensor_img.to(device)
    model.to(device)

    result_time = []

    for i in tqdm(range(500)):
        # print(i)
        start = time.time()
        pred = model([tensor_img])
        end = time.time()
        t1 = end-start
        result_time.append(t1)
        

    print("predict time:", result_time)

    df = pd.DataFrame()
    df[csvfile] = result_time
    df.to_csv(csvfile+'.csv', index=False)
    return

for obj in test_cases:
    img_path, csvfile = obj.img_path, obj.csvfile
    print('img_path:', img_path)
    print('csvfile:', csvfile)
    test(img_path, csvfile)
print('done')



# boxes = pred[0]['boxes']
# labels = pred[0]['labels']
# scores = pred[0]['scores']

# threshold = 0.5

# pred_index = scores > threshold

# boxes = boxes[pred_index]
# labels = labels[pred_index]
# labels = [names.get(str(idx.item())) for idx in labels]
# img = torch.from_numpy(np_img).permute(2, 0, 1)

# result = torchvision.utils.draw_bounding_boxes(
#                         img,
#                         boxes,
#                         labels)

# plt.figure(figsize=(8, 10))
# plt.imshow(result.permute(1,2,0).numpy())
# # plt.savefig('2007_000027.jpg', dpi=300)
# plt.show()


