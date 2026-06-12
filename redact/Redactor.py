import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from collections import Counter

from activitygen.merge.element import Element


class Redactor:
    # Type of redaction as param
    def __init__(self):
        pass

    def redact(self, img, elements, type="block"):
        if type == "block":
            return self.block(img, elements)
        elif type == "label":
            return self.substitute_by_label(img, elements)

    # Block area with average color
    def block(self, img, elements, color=(0, 0, 0)):
        for element in elements:
            cv2.rectangle(
                img, (element.x1, element.y1), (element.x2, element.y2), color, -1
            )

    # Substitute background area with mode color and text with second most frequent color (-> probably text color)
    def substitute_by_label(self, img, elements: List[Element]):
        og_image = img.copy()
        for number, element in enumerate(elements):
            # get 2 most frequent colors in element region
            region = og_image[element.y1 : element.y2, element.x1 : element.x2]

            region_1d = np.reshape(region, (-1, 3))
            region_1d = [tuple(bgr) for bgr in region_1d]

            most_common = Counter(region_1d).most_common(10)

            # Redact original with mode color
            if len(most_common) == 0:
                color_bg = (255, 255, 255)
            else:
                color_bg = tuple(int(c) for c in most_common[0][0])
            cv2.rectangle(
                img,
                (element.x1, element.y1),
                (element.x2, element.y2),
                color_bg,
                -1,
            )
            # region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            # plt.imshow(region_rgb)
            # plt.show(block=True)

            # Put label text with second most common color
            color_text = (0, 0, 0)  # default color black
            for common in most_common[1:]:
                color = tuple(int(c) for c in common[0])
                color_dist = np.sqrt(
                    np.sum((np.array(color_bg) - np.array(color)) ** 2)
                ) / np.sqrt(3 * 255**2)
                if color_dist > 0.4:
                    color_text = color
                    break

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Rough text size calculation
            font_scale = self.calculate_text_scale(
                element.substitution, region.shape, font
            )
            thickness = math.ceil(
                min(region.shape[0], region.shape[1]) * 2 * (font_scale / 100)
            )
            TEXT_Y_OFFSET_SCALE = 0.25
            TEXT_X_OFFSET_SCALE = 0.05
            y_offset = int((element.y2 - element.y1) * TEXT_Y_OFFSET_SCALE)
            x_offset = int((element.x2 - element.x1) * TEXT_X_OFFSET_SCALE)

            cv2.putText(
                img,
                element.substitution,
                (element.x1 + x_offset, element.y2 - y_offset),
                font,
                font_scale,
                color_text,
                thickness,
                cv2.LINE_AA,
            )

    # Binary search for scale of subsitution text
    def calculate_text_scale(self, text, area, font):
        height = area[0]
        width = area[1]
        min_scale = 0.1
        max_scale = 10.0
        best_scale = min_scale

        while max_scale - min_scale > 0.01:
            mid_scale = (min_scale + max_scale) / 2
            text_size, _ = cv2.getTextSize(text, font, mid_scale, 1)
            text_width, text_height = text_size

            if text_width <= width and text_height <= height:
                best_scale = mid_scale
                min_scale = mid_scale
            else:
                max_scale = mid_scale

        return best_scale
