# Diabetic Retinopathy - one shot

# Import modules
import pandas as pd
import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import os

# Load Data
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
train_image_folder = 'dataset/train_images'
test_image_folder = 'dataset/test_images'

# Check data load
print("Train Data:")
print(train_df.head())
print("\nTest Data:")
print(test_df.head())

# Max-Min resolution img
smallest_width = float('inf')
smallest_height = float('inf')
largest_width = 0
largest_height = 0

for filename in os.listdir(train_image_folder):
    if filename.endswith('.png'):
        img = cv2.imread(os.path.join(train_image_folder, filename))
        height, width, _ = img.shape
        smallest_width = min(smallest_width, width)
        smallest_height = min(smallest_height, height)
        largest_width = max(largest_width, width)
        largest_height = max(largest_height, height)

    print(
        "Smallest image resolution (width x height):", smallest_width, "x", smallest_height, '\n',
        "Largest image resolution (width x height):", largest_width, "x", largest_height,
          end='\r'
    )
print()
smallest_width = float('inf')
smallest_height = float('inf')
largest_width = 0
largest_height = 0

for filename in os.listdir(test_image_folder):
    if filename.endswith('.png'):
        img = cv2.imread(os.path.join(test_image_folder, filename))
        height, width, _ = img.shape
        smallest_width = min(smallest_width, width)
        smallest_height = min(smallest_height, height)
        largest_width = max(largest_width, width)
        largest_height = max(largest_height, height)

print("Smallest image resolution (width x height):", smallest_width, "x", smallest_height)
print("Largest image resolution (width x height):", largest_width, "x", largest_height)
