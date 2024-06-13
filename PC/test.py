import fiftyone as fo
from fiftyone import ViewField as F

import matplotlib.pyplot as plt
import pandas as pd



def func1():
    # import fiftyone.zoo as foz
    # dataset = foz.load_zoo_dataset("coco-2017")

    # The directory containing the source images
    data_path = "D:/code/python/dataset/customized-coco-2017/val2017"

    # The path to the COCO labels JSON file
    labels_path = "D:/code/python/dataset/customized-coco-2017/annotations/instances_val2017.json"


    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
    )


    session = fo.launch_app(dataset)
    session.wait()

def func2():


    # Load the CSV file
    file_path = 'result-all-500 - 3.csv'
    data = pd.read_csv(file_path)

    # Calculate the mean for each column
    means = data.mean()

    # Grouping columns for the plot
    groups = {
        '1920x1080': ['faster-rcnn-pc-1920x1080', 'yolov8n-pc-1920x1080' ],
        '1280x720': ['faster-rcnn-pc-1280x720', 'yolov8n-pc-1280x720'],
        '720x480': ['faster-rcnn-pc-720x480', 'yolov8n-pc-720x480']
    }

    # Plotting the means
    fig, ax = plt.subplots()
    width = 0.2  # the width of the bars

    for i, (group_name, columns) in enumerate(groups.items()):
        group_means = means[columns]
        ax.bar([x + i * width for x in range(len(group_means))], group_means, width, label=group_name)

    # Adding some text for labels, title and axes ticks
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean values by Resolution and Model')
    ax.set_xticks([x + width / 2 for x in range(len(group_means))])
    ax.set_xticklabels(['faster-rcnn-pc', 'yolov8n-pc'])
    ax.legend()

    # Display the plot
    plt.show()

    return

def func3():
    # 
    data = pd.read_csv('result-pc.csv', header=None, names=['faster r-cnn', 'yolov8n'])

    # 
    fig, ax = plt.subplots(figsize=(10, 6))

    # 
    ax.plot(data['faster r-cnn'], label='faster r-cnn')
    ax.plot(data['yolov8n'], label='yolov8n')

    # 
    ax.set_title('Performance Comparison')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Performance Score')

    # 
    ax.legend()

    # 
    plt.show()

func2()

print("done")