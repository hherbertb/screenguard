import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import easyocr
import cv2
import time
from datetime import datetime
from os.path import join as pjoin
from redact.classification.LocalLLMClassifier import LocalLLMClassifier
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

# Init models
reader = easyocr.Reader(["en"])
detector = TextDetector(output_root=output_root, model=reader)
if len(sys.argv) > 2:
    model_name = sys.argv[2]
else:
    model_name = "qwen2.5:72b"
llm_classifier = LocalLLMClassifier(model_name=model_name)
redactor = Redactor()

# Prepare component detection
compo_detector = CompoDetector(output_root)


start = time.time()

for img in images:
    img_path = input_path + img

    # Detect text
    detector.detect_text(img_path)

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
    if len(sys.argv) > 1:
        prompt_type = sys.argv[1]
    else:
        prompt_type = "ShortFewShot"
    print(f"Using prompt {prompt_type}.")
    print(f"Using prompt {model_name}.")
    private_elements = llm_classifier.filter(elements, prompt_type=prompt_type)

    # Redact private elements
    img = cv2.imread(img_path)
    redactor.redact(img, private_elements, "label")

    # Output final image
    cv2.imwrite(redacted_root + img_name + ".png", img)
    time.sleep(60)


    


end = time.time()
print(f"Runtime was {end - start}s.")
# Terminate Ollama model
os.system(f"ollama stop {model_name}")
