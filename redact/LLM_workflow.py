import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import easyocr
import cv2
import time
import matplotlib.pyplot as plt

from datetime import datetime
from os.path import join as pjoin
from redact.classification.LLMPrivacyClassifier import LLMPrivacyClassifier
from redact.detection.TextDetector import TextDetector
from redact.Redactor import Redactor
from redact.Utils import convert_data_to_elements
from redact.classification.Context import *

# ActivityGen imports for component detection
from activitygen.config.parameter_config import CONFIG_PARSER, load_config

load_config("./configs/main.cfg")  # ! Load before, otherwise crash
from activitygen.compo_detector.detector import CompoDetector
from activitygen.compo_detector.Utils import resize_height_by_longest_edge
from activitygen.merge import merger


# Prepare input and output
input_path = "input/"
images = os.listdir("input/")
output_root = "output/" + datetime.now().strftime("%H-%M-%S__%d-%m-%Y") + "/"
os.mkdir(output_root)
redacted_root = pjoin(output_root, "redacted/")
os.mkdir(redacted_root)
classification_dir = redacted_root + "jsons/"
os.mkdir(classification_dir)
blocked_dir = redacted_root + "blocked/"
os.mkdir(blocked_dir)

# Init models
reader = easyocr.Reader(["en"], recog_network="best_accuracy")
detector = TextDetector(output_root=output_root, model=reader)
llm_classifier = LLMPrivacyClassifier()
redactor = Redactor()

# Prepare component detection
compo_detector = CompoDetector(output_root)


performance = {}
start = time.time()
for img_label in images:
    print(f"----------------------------------------------------IMAGE:{img_label}------------------------------------------------------")
    img_path = input_path + img_label

    # Detect text
    detector.detect_text(img_path)

    # Filter text in taskbar
    

    # Detect components
    resized_height = resize_height_by_longest_edge(
        img_path, resize_length=CONFIG_PARSER.getint("MAIN", "resize_length")
    )
    compo_detector.detect_compos(
        input_img_path=img_path, resize_by_height=resized_height, show=False
    )

    # Merge components and OCR
    img_name = img_path.split("/")[-1][:-4]

    compo_path = pjoin(output_root, "compo", str(img_name) + ".json")
    ocr_path = pjoin(output_root, "ocr", str(img_name) + ".json") 

    img_resize_shape, resized_image, data = merger.merge(
        img_path,
        compo_path,
        ocr_path,
        merge_root=pjoin(output_root, "merge"),
        classifier=None,
        show=False,
    )

    # Convert merged text and components to Element objects
    merge_path = pjoin(output_root, "merge", str(img_name) + ".json")
    elements = convert_data_to_elements(merge_path)

    # Add context to elements
    add_nearest_element_context(elements, max_distance=50)
    
    # Retrieve privacy relevant elements
    classification_file = classification_dir + img_name + ".json"
    open(classification_file, 'a').close() # create file

    prompt_type="main2" # CHANGE
    print(f"Using prompt type {prompt_type}.")
    # CHANGED BUNDLE SIZE TO 3
    private_elements = llm_classifier.filter(elements, classification_file, prompt_type=prompt_type, bundle_size=3, use_context=True) # CHANGE

    # Redact private elements
    img_label = cv2.imread(img_path)
    img_block = img_label.copy()

    redactor.redact(img_label, private_elements, "label")
    redactor.redact(img_block, private_elements, "block")

    # Output final image
    cv2.imwrite(redacted_root + img_name + ".png", img_label)
    cv2.imwrite(blocked_dir + img_name + "_block.png", img_block)


    


end = time.time()
print(f"Runtime was {end - start}s.")
