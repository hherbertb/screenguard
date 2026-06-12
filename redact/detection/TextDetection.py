# Achieve independence from underlying OCR lib and enable to add context
import numpy as np

class TextDetection:
    # bbox: list of 4 tuples, upperleft,  
    def __init__(self, id, bbox, text, confidence):
        self.id = id
        # round bbox to int
        self.bbox = [(int(p[0]), int(p[1])) for p in bbox]

        self.text = text
        self.confidence = confidence
        self.ul = self.bbox[0]
        self.ur = self.bbox[1]
        self.lr = self.bbox[2]
        self.ll = self.bbox[3]
        

def convert_easyocrs_to_detection(ocr_detections):
    detections = []
    ctr = 0
    for ocr in ocr_detections:
        detection = TextDetection(ctr, ocr[0], ocr[1], ocr[2])
        detections.append(detection)
        ctr += 1
    return detections
