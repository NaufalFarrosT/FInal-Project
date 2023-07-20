# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import time
import cv2

import PIL
from PIL import Image

def print_hi(name):

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Image
    im = '1.jpg'


    # Inference
    results = model(im)

    results.show()


    #print(results)
    #print(results.pandas().xyxy[0])

    #print(results.xyxy[0])

    #TEST WITH PREPOCESSING
    image = Image.open('1.jpg')
    newImage = image.resize((640, 448))
    newImage.save('1-Prepocessing-PILLOW.jpg')

    results2 = model(newImage)

    print(results2)
    print(results2.pandas().xyxy[0])

    # set the device we will be using to run the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    # load the list of categories in the COCO dataset and then generate a
    # set of bounding box colors for each class




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
