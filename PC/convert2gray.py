

import cv2
import os
import shutil

def convertOneImg():
    img = cv2.imread('./data/test/car-and-bus.jpg')
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Save the grayscale image in dir2
    cv2.imwrite('./data/test/car-and-bus_gray.jpg', gray)

def convertImg():
    # Define the directories
    dir1 = 'D:/code/python/datasets/customized-coco-2017/images/train2017'
    dir2 = 'D:/code/python/datasets/coco-2017-gray/images/train2017'

    # Loop through all files in dir1
    files = os.listdir(dir1)
    for i,filename in enumerate(files):
        if i>=1000:
            break
        # Check if the file is an image
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            # Read the image
            img = cv2.imread(os.path.join(dir1, filename))
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Save the grayscale image in dir2
            cv2.imwrite(os.path.join(dir2, filename), gray)


def copyLabels():
    # Define the directories
    dir1 = 'D:/code/python/datasets/customized-coco-2017/labels/train2017'
    dir2 = 'D:/code/python/datasets/coco-2017-gray/labels/train2017'
    files = os.listdir(dir1)
    for i,filename in enumerate(files):
        if i>=1000:
            break
        shutil.copy(os.path.join(dir1, filename), dir2)

    return

# convertImg()

# copyLabels()
convertOneImg()


