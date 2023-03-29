# -*- coding: utf-8 -*-
"""
This is where utility code and helper functions are located
"""

import albumentations as A
import cv2
import numpy as np

# We will be defining training and validation augmentation here
from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES as classes

# this class keeps track of the training and validation loss values
# and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
        
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
        
    def reset(self):
        self.current_total = 0.0
        self.iteration = 0.0
        
# Handles data loading, as different images may have different
# number of objects. Also handles different sized tensors (bounding boxes)
def collate_fn(batch):
    return tuple(zip(*batch))

""" 
    For the transformations below, albumentations handles the orientation of the corresponding
    bounding boxes when applying the augmentation, significantly reducing manual coding
"""

# define the training transforms
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format':'pascal_voc',
        'label_fields': ['labels']
    })
        
# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
        
# If VISUALIZE_TRANSFORMED_IMAGE in config.py is True, the following will run
# This function will show the transformed images in order to check if the transformed
# image, along with corresponding labels, are correct or not
def show_transformed_image(train_loader):
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
            cv2.imshow('Transformed Image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()