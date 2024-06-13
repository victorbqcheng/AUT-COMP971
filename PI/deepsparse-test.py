from deepsparse import Pipeline
import time
import cv2
import pandas as pd
from tqdm import tqdm

# Specify the path to your YOLOv8 ONNX model
model_path = "./yolov8n.onnx"
# task = "yolo"
# stub = "zoo:yolov5-n-voc_coco-pruned30.4block_quantized"
# yolo_pipeline = Pipeline.create(task, model_path=stub, batch_size=1)
# Set up the DeepSparse Pipeline
yolo_pipeline = Pipeline.create(
    task="yolov8",
    model_path=model_path,
    batch_size=1
)

# Run the model on your images
img_path = "./data/test/car-and-bus.jpg"
img_path = "./data/test/car1.jpg"
# img_path = "./data/test/car2.jpg"

class Obj:
    def __init__(self, img_path, csvfile) -> None:
        self.img_path = img_path
        self.csvfile = csvfile
        pass

test_cases = [
    Obj('./data/test/car-1280x720.jpg', 'yolov8n-sparsified-1280x720'),
    Obj('./data/test/car-1920x1080.jpg', 'yolov8n-sparsified-1920x1080'),
    Obj('./data/test/car-720x480.jpg', 'yolov8n-sparsified-720x480'),
    Obj('./data/test/car-640x426.jpg', 'yolov8n-sparsified-640x426'),
    Obj('./data/test/car-360x240.jpg', 'yolov8n-sparsified-360x240'),
    ]

def test(img_path, csvfile):
    img = cv2.imread(img_path)
    images = [img]

    result_time = []
    for i in tqdm(range(500)):
        start = time.time()
        pipeline_outputs = yolo_pipeline(images=images)
        end = time.time()
        result_time.append(end-start)

    # print('predict time:', result_time)

    df = pd.DataFrame([])
    # df = pd.read_csv('./result.csv')
    df[csvfile] = result_time
    df.to_csv(csvfile + '.csv', index=False)

    return

for obj in test_cases:
    img_path, csvfile = obj.img_path, obj.csvfile
    print('img_path:', img_path)
    print('csvfile:', csvfile)
    test(img_path, csvfile)
print('done')




