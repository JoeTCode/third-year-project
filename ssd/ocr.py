from PIL import Image
def read_numberplate(pil_image, bbox):
    img = Image.open(pil_image)
    x_min, y_min, x_max, y_max = bbox
    cropped_img = img.crop((x_min, y_min, x_max, y_max))
    # cropped_img.show()
