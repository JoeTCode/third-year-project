from ssd.custom_coco_dataset_loader import AnprCocoDataset, Resize, ToTensor
from ssd.config.config import COCO_TRAIN_ROOT, COCO_TRAIN_ANNOTATIONS_FILE, BATCH_SIZE
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from ssd.train_model import collate_fn


transform = transforms.Compose([
    Resize((300, 300)),
    ToTensor()
])

dataset = AnprCocoDataset(
    train_annotations_file_path=COCO_TRAIN_ANNOTATIONS_FILE,
    train_images_root=COCO_TRAIN_ROOT,
    transform=transform
)


def test_coco_dataloader(end_test, image_shape, has_background_images, drop_last, batch_size=BATCH_SIZE):
    assert isinstance(image_shape, tuple) and len(image_shape) == 3, "Please provide tuple in the form (C, H, W)"
    channels, height, width = image_shape

    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=drop_last
    )

    for i_batch, sample_batched in enumerate(train_dataloader):
        img_batch, annotations_batch = sample_batched

        if not drop_last:
            # Check if images in images batch tensor is correct dimensions (e.g. C:3, H:300, W:300)
            assert img_batch.shape[1:] == (channels, height, width)
        else:
            assert img_batch.shape == (batch_size, channels, height, width)
            assert len(annotations_batch) == batch_size

        for annotation in annotations_batch:
            assert list(annotation.keys()) == ['image_id', 'boxes', 'labels']
            assert isinstance(annotation['image_id'], torch.Tensor)

            if not has_background_images:
                assert len(annotation['boxes'] >= 1) # Check if there is at least one detection for the image

            assert isinstance(annotation['boxes'], torch.Tensor)
            assert isinstance(annotation['labels'], torch.Tensor)

            if isinstance(annotation['labels'], torch.Tensor):
               assert annotation['labels'].shape[0] == len(annotation['boxes'])

        if i_batch == end_test:
            break


test_coco_dataloader(5, (3, 300, 300), False, True)
