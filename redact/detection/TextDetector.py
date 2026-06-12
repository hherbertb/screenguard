import cv2
import json

from os.path import join as pjoin
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from redact.Utils import build_directory
from redact.detection.TextDetection import convert_easyocrs_to_detection, TextDetection


class TextDetector:
    def __init__(self, output_root, model):
        self.model = model
        self.output_root = output_root

    def save_detection_json(self, file_path, detections, img_shape):
        with open(file_path, "w") as f_out:
            output = {"img_shape": img_shape, "texts": []}
            for detection in detections:
                c = {
                    "id": detection.id,
                    "content": detection.text,
                    "x1": detection.ul[0],
                    "y1": detection.ul[1],
                    "x2": detection.lr[0],
                    "y2": detection.lr[1],
                    "width": detection.lr[0] - detection.ul[0],
                    "height": detection.lr[1] - detection.ul[1],
                }
                output["texts"].append(c)
            json.dump(output, f_out, indent=4, ensure_ascii=False)

    # Detects text on cv2 image
    def detect_text(self, img_path):
        # Make output folder for detection
        ocr_root = build_directory(pjoin(self.output_root, "ocr"))
        name = img_path.split("/")[-1][:-4]
        img = cv2.imread(img_path)

        # Detect individual words
        ocr_detections = self.model.readtext(
            img,
            min_size=1,
            mag_ratio=2,
            low_text=0.4,
            link_threshold=0.01,
            text_threshold=0.5,
            slope_ths=0,
        )

        # convert detections to detection class
        detections = convert_easyocrs_to_detection(ocr_detections)


        # Visualize detections
        img = self.visualize_detections(img, detections, scale=0.65)

        # Show image
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()

        # Store image
        cv2.imwrite(pjoin(ocr_root, name + ".png"), img)

        # save detection as json for merging later
        self.save_detection_json(
            pjoin(ocr_root, name + ".json"),
            detections,
            img.shape,
        )
        return detections

    def visualize_detections(self, img, detections, scale=1.0, threshold=0):
        for detection in detections:
            if detection.confidence > threshold:
                cv2.rectangle(img, detection.ul, detection.lr, (0, 255, 0), 2)

                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                font = ImageFont.truetype("configs/fonts/Arial.ttf", int(0.65 * scale * 30))

                draw = ImageDraw.Draw(img_pil)
                draw.text((detection.ul[0], detection.ul[1] - 10), detection.text, font=font, fill=(255, 0, 0))

                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return img

    def sanitize_detections(detections: TextDetection):
        pass
