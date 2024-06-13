from deepsparse.pipeline import Pipeline

task = "yolo"
stub = "zoo:yolov8-n-coco-pruned48_quantized"
pipeline = Pipeline.create(task, model_path=stub, batch_size=1)
print(pipeline)