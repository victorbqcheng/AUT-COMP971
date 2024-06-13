from cProfile import Profile
from pstats import SortKey, Stats

from ultralytics import YOLO
from deepsparse import Pipeline


pt_file = 'yolov8n.pt'
model = YOLO(pt_file)


img = "./data/test/car-and-bus.jpg"


results = model(img, verbose=False)

with Profile() as profile:
    print(f"{model(img, verbose=False)=}")
    (
        Stats(profile)
        # .strip_dirs()
        .sort_stats(SortKey.TIME)
        .print_stats()
    )



# profiler = Profile()

# model_path = "./yolov8n.onnx"
# yolo_pipeline = Pipeline.create(
#     task="yolov8",
#     model_path=model_path,
#     batch_size=1
# )
# images = [img]

# profiler.enable()
# pipeline_outputs = yolo_pipeline(images=images)
# Stats(profiler).strip_dirs().sort_stats(SortKey.TIME).print_stats()
