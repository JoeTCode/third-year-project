from ssd.custom_yolo_dataset_loader import AnprYoloDataset, Resize, ToTensor
from config import config
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch


transform = transforms.Compose([
    Resize((300, 300)),
    ToTensor()
])

anpr_yolo_dataset = AnprYoloDataset(
        annotations_root=config.TRAIN_ANNOTATIONS_ROOT,
        images_root=config.TRAIN_IMAGES_ROOT,
        transform=transform
    )


def collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])  # Image tensor
        targets.append(sample[1])  # List of annotations (dictionary)

    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, targets  # Targets remain as a list of dicts (not a tensor)


def test_yolo_dataloader(end_test, image_shape, has_background_images, drop_last, batch_size=config.BATCH_SIZE):
    assert isinstance(image_shape, tuple) and len(image_shape) == 3, "Please provide tuple in the form (C, H, W)"
    channels, height, width = image_shape

    train_dataloader = DataLoader(
        anpr_yolo_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=drop_last
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


#test_yolo_dataloader(5, (3, 300, 300), False, True)


train_dataloader = DataLoader(
    anpr_yolo_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn
)

# In the form (image_tensor, {'image_id': tensor(1), 'boxes': tensor([[bbox], [bbox]), 'labels': tensor([1, 1])})
image_tensor1, annotations1 = anpr_yolo_dataset[0]
image_tensor2, annotations2 = anpr_yolo_dataset[1]
image_tensor3, annotations3 = anpr_yolo_dataset[2]
image_tensor4, annotations4 = anpr_yolo_dataset[3]

pil1 = transforms.ToPILImage()(image_tensor1)
pil2 = transforms.ToPILImage()(image_tensor2)
pil3 = transforms.ToPILImage()(image_tensor3)
pil4 = transforms.ToPILImage()(image_tensor4)





# for i_batch, sample_batched in enumerate(train_dataloader):
#     img_batch, annotations_batch = sample_batched
#     print(img_batch[0])
#     break
