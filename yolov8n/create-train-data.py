import fiftyone as fo
import fiftyone.zoo as foz

# setting dataset name
dataset_name = "open-images-v7"

# categories to download from openimages
classes = ["Vehicle registration plate"]

# load the dataset with the specified category
dataset = foz.load_zoo_dataset(
    dataset_name,
    split="train",  # dataset type e.g. train, validation, test, etc.
    label_types=["detections"],  # type of annotation e.g. detections, segmentation etc.
    classes=classes,  # category types e.g. person, car, vehicle registration plate etc.
    max_samples=4000,  # limit the number of samples
)

