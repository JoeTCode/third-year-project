import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.

ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = './img.png'

img = Image.open(img_path)
# img.show()
np_image = np.array(img)
bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
result = ocr.ocr(bgr, cls=True)

# if result is None or len(result) == 0 or result[0] is None:
#     print("No text detected.")
print(result)
print(f'result[0] {result[0]}')
if result is not None:
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line[1][0])