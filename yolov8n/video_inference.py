# https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/

import os
import cv2
import time

from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
from yolov8n.load_model_and_infer import draw_bbox

video_path = '/Users/joe/Downloads/2103099-uhd_3840_2160_30fps.mp4'
model = YOLO("/Users/joe/Code/third-year-project/ANPR/yolov8n/weights/run9_best.pt")  # pretrained

def extract_frames(video_path, frames_save_directory):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)

    count = 0
    success = True
    while success:
        success, frame = cap.read()
        if success:
            filename = f"frame-{count}.jpg"
            cv2.imwrite(os.path.join(frames_save_directory, filename), frame)
            count += 1

    end_time = time.time()
    # Elapsed time: 134.95 seconds (to split 30 second 30 fps video - 1800 frames)
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")


def frame_inference(frames_directory, model, output_video_num_frames):
    # Run inference, returns generator due to stream=True. Saves memory.
    #results = model.predict(source=frames_directory, stream=True, show=False, conf=0.4, verbose=False, save=False)
    images = os.listdir(frames_directory)
    images = [img for img in images if not img.startswith('.')] # remove hidden files like '.DS_Store' (for mac)
    images = sorted(images, key=lambda x: int(x.split('-')[1].split('.')[0]))

    if output_video_num_frames >= len(images):
        output_video_num_frames = len(images) - 1

    first_image_path = images[0]
    first_image = Image.open(os.path.join(frames_directory, images[0]))
    width, height = first_image.size
    first_image.close()

    # Video writer to create .mp4 file
    video = cv2.VideoWriter('./video/anpr_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))


    for i, image_filepath in enumerate(images):
        image_path = os.path.join(frames_directory, image_filepath)
        result = model.predict(source=image_path, show=False, conf=0.4, verbose=False, save=False)

        image = Image.open(image_path)
        for j, r in enumerate(result):
            draw = ImageDraw.Draw(image)
            predicted_bboxes = r.boxes.xyxy
            print(predicted_bboxes)
            predicted_scores = r.boxes.conf
            print(predicted_scores)

            for k, predicted_bbox in enumerate(predicted_bboxes):
                predicted_score = predicted_scores[k]
                draw_bbox(image, draw, predicted_bbox, predicted_score, show_steps=False)

        np_image = np.array(image)

        video.write(np_image)

        if i == output_video_num_frames:
            return video


if __name__ == "__main__":
    start = time.time()
    output_frames = 100
    video = frame_inference('./frames', model, output_frames)

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print(f"Video generated successfully in {time.time() - start:.2f} seconds, with {output_frames} frame(s)!")

# Video generated successfully in 78.71 seconds, with 100 frame(s)!
