# -*- coding: utf-8 -*-
"""
All training configurations are stored here to reduce
excessive fiddling with different parts of the pipeline
"""

import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 100 # number of epochs to train for

DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

# training images and XML files directory
TRAIN_DIR = '../Microcontroller Detection/train/'
# validation images and XML files directory
VALID_DIR = '../Microcontroller Detection/test/'

# classes: 0 index is reserved for background
# PyTorch Faster RCNN expects the background class
CLASSES = [
    'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
]
NUM_CLASSES = len(CLASSES)

# whether to visualize images after clearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs