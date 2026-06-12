import os
import json
import math
import numpy as np
from activitygen.merge.element import Element
from shapely.geometry import Polygon
from typing import Dict


def build_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def convert_data_to_elements(merge_path):
    merge_json = json.load(open(merge_path, "r"))

    # load merged elements
    elements = {}
    for merged_compo in merge_json["compos"]:
        element = Element(
            merged_compo["id"],
            (
                merged_compo["position"]["x1"],
                merged_compo["position"]["y1"],
                merged_compo["position"]["x2"],
                merged_compo["position"]["y2"],
            ),
            merged_compo["class"],
        )
        # Add optional info
        element.text_content = merged_compo.get("text_content")
        element.children = (
            merged_compo.get("children") if merged_compo.get("children") else []
        )
        element.parent_id = merged_compo.get("parent")

        # add element to dict
        elements[merged_compo["id"]] = element

    return elements


def distance(element: Element, other: Element, method="closest"):
    # Euclidean distance between centers of rectangles
    if method == "center":
        center1 = (
            element.x1 + (element.x2 - element.x1) / 2,
            element.y1 + (element.y2 - element.y1) / 2,
        )
        center2 = (
            other.x1 + (other.x2 - other.x1) / 2,
            other.y1 + (other.y2 - other.y1) / 2,
        )

        euclid_distance = math.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )

        return euclid_distance
    # Shortest distance between both rectangles borders. 0 if intersected or contained
    elif method == "closest":
        rect1 = Polygon(
            (
                (element.x1, element.y1),
                (element.x1, element.y2),
                (element.x2, element.y2),
                (element.x2, element.y1),
                (element.x1, element.y1),
            )
        )
        rect2 = Polygon(
            (
                (other.x1, other.y1),
                (other.x1, other.y2),
                (other.x2, other.y2),
                (other.x2, other.y1),
                (other.x1, other.y1),
            )
        )

        minimal_distance = rect1.distance(rect2)
        return minimal_distance


def refine_elements_with_tables(elements: Dict[int, Element], tables):
    pass
