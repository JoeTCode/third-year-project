import fiftyone.zoo as foz

dataset_name = "open-images-v7"
# categories to download from openimages
classes = ["Vehicle registration plate"]

# Load a separate validation dataset
val_dataset = foz.load_zoo_dataset(
    dataset_name,
    split="validation",  # Specify the validation split
    label_types=["detections"],
    classes=classes,
    max_samples=1000,  # Limit to a smaller number of samples for validation
)
