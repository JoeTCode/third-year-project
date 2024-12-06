import os
import pandas as pd

annotations_path = "/Users/joe/fiftyone/open-images-v7/train/labels"
images_path = '/Users/joe/fiftyone/open-images-v7/train/images'

annos = os.listdir(annotations_path)
images = os.listdir(images_path)

# Extract the base filenames without extensions
anno_bases = {os.path.splitext(anno)[0] for anno in annos}
image_bases = {os.path.splitext(img)[0] for img in images}

# Check if every image has a corresponding annotation file
missing_annotations = image_bases - anno_bases
missing_images = anno_bases - image_bases

if missing_annotations:
    print(f"Missing annotations for images: {missing_annotations}")
else:
    print("All images have corresponding annotations.")

if missing_images:
    print(f"Missing images for annotations: {missing_images}")
else:
    print("All annotations have corresponding images.")

for row in annos:
    print(row)