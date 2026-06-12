import pandas as pd
import numpy as np

from typing import Dict
from scipy.spatial.distance import pdist

from activitygen.merge.element import Element
from redact.Utils import distance

class Context:
    def __init__(self, element_id: Element, distance):
        self.element_id = element_id
        self.distance = distance



def add_nearest_element_context(elements: Dict[int, Element], max_distance=50):
    texts = {
            id: element
            for id, element in elements.items()
            if element.category == "Text"
    }
    distances = compute_distances(texts)

    for id, text in texts.items():
        text.context = find_nearest_elements(text, distances, max_distance)
    print("")



def compute_distances(elements: Dict[int, Element]):
    ids = list(elements.keys())
    n = len(elements)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist = distance(elements[ids[i]], elements[ids[j]])
            distance_matrix[i,j] = dist
            distance_matrix[j,i] = dist

    distance_df = pd.DataFrame(distance_matrix, index=ids, columns=ids)
    return distance_df

def find_nearest_elements(element: Element, distance_df: pd.DataFrame, max_distance=50, n=None):
    distances = distance_df.loc[element.id]
    # near elements sorted by distance
    near_elements = distances[distances < max_distance].sort_values().index.tolist()
    near_elements_distances = distances[distances < max_distance].sort_values().tolist()
    near_elements.remove(element.id) # self
    near_elements_distances.pop(0) # self
    if n is not None and n <= len(near_elements):
        near_elements[:n]
        near_elements_distances[:n]

    result = []
    for i, near in enumerate(near_elements):
        result.append(Context(near, near_elements_distances[i]))

    
    return result



