from PIL import ImageDraw, Image
from torchvision import transforms
import os
import time
import easyocr
import numpy as np

# Initialise EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

def read_numberplate(pil_image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cropped_img = pil_image.crop((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
    return cropped_img

def map_bbox_to_image(image, target_bboxes, predicted_bboxes, scores, save_directory):
    image = transforms.ToPILImage()(image)  # Convert from tensor to PIL image for visualization
    draw = ImageDraw.Draw(image)
    timestamp = time.time()
    # Draw predicted bboxes in red
    for i in range(predicted_bboxes.shape[0]):
        x_min, y_min, x_max, y_max = predicted_bboxes[i]
        text = str(round(scores[i].item(), 2))  # Gets score for bounding box in the desired format.
        # Gets location of text so a rectangle can be mapped under it
        xmin, ymin, xmax, ymax = draw.textbbox(xy=(x_min, y_min - 10), text=text)

        # Draw confidence score in white (along with a red background) above the predicted bbox
        draw.rectangle([xmin, ymin, xmax, ymax], fill="red")
        draw.text(xy=(x_min, y_min-10), text=text, fill="white")

        # Draws the predicted bounding box outline in red
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=1)
        cropped_image = read_numberplate(image, [x_min, y_min, x_max, y_max])
        np_cropped_image = np.array(cropped_image)
        detections = reader.readtext(np_cropped_image)
        print(detections)

    # Draw target (actual) bbox in green
    for i in range(target_bboxes.shape[0]):
        x_min, y_min, x_max, y_max = target_bboxes[i]
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=1)

    # Show image
    #image.show()

    # filepath = os.path.join(save_directory, f'{timestamp}.png')
    # image.save(filepath)
