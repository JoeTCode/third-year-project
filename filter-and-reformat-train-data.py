import pandas as pd
import os

annotations_path = "/Users/joe/fiftyone/open-images-v7/train/labels/detections.csv"
images_path = '/Users/joe/fiftyone/open-images-v7/train/images'

df = pd.read_csv(annotations_path)
columns = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
filtered_annotations = df[df["LabelName"] == '/m/01jfm_'][columns]

# check if the annotations match the downloaded images
downloaded_images = [image.split('.')[0] for image in os.listdir(images_path)]
filtered_annotations = filtered_annotations[filtered_annotations["ImageID"].isin(downloaded_images)]

# MID: /m/01jfm_ is License plate
class_map = {'/m/01jfm_': 0}
annotations_grouped = filtered_annotations.groupby("ImageID")
for image in downloaded_images:
    if image in annotations_grouped.groups:
        # get rows corresponding to this image
        image_annotations = annotations_grouped.get_group(image)
        with open(f"/Users/joe/fiftyone/open-images-v7/train/labels/{image}.txt", "w") as f:
            for _, row in image_annotations.iterrows():
                class_id = class_map[row["LabelName"]]
                center_x = (row["XMin"] + row["XMax"]) / 2
                center_y = (row["YMin"] + row["YMax"]) / 2
                width = row["XMax"] - row["XMin"]
                height = row["YMax"] - row["YMin"]
                f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
