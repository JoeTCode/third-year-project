import albumentations as A
import cv2
import os
from config import config
import numpy as np

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.3),
    A.HorizontalFlip(p=0.3),
    A.Affine(
        translate_percent=(-0.1, 0.1), scale=(0.5, 1.5), rotate=(-5, 5), p=0.7
    ),
    A.Perspective(scale=(0.05, 1), keep_size=True, p=0.4),
    A.Resize(300, 300),
])


image_files = [image for image in os.listdir(config.TEST_IMAGES_ROOT)]
image_filename = '87b4f1202cd06440_jpg.rf.3ee32f70ec974208471b0fb9674710ff.jpg'
image_index = image_files.index(image_filename)
image_path = os.path.join(config.TEST_IMAGES_ROOT, image_files[image_index])
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
augmented = transform(image=image)
augmented_image = augmented["image"]
cv2.imshow('original', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow('augmented', cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()