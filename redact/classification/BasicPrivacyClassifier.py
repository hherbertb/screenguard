import re
import copy
import json

from typing import Dict
import importlib.resources
from symspellpy import SymSpell, Verbosity

from redact.Utils import distance

from activitygen.merge.element import Element


class BasicPrivacyClassifier:
    def __init__(self):
        self.reg_exs = [
            # phone
            r"^(\+?\d{1,2}\s?)?(\()?(\d{3})(?(2)\))[\s.-]?\d{3}[\s.-]?\d{4}$",
            # mail (with optional . and whitespaces)
            r"^\s*[a-zA-Z0-9._%+-]\s*(\s*[a-zA-Z0-9._%+-]\s*)*@\s*[a-zA-Z0-9-]\s*(\s*[a-zA-Z0-9-]\s*)*(\s*\.\s*[a-zA-Z]{2,})?\s*$",
            # aggressive regex for links
            r".*https?.*\/.*",
            # aggressive regex for ids
            r"^#[ ]*\d+$",
            # money amounts
            r"(^\d+(\.\d+)?[ ]*\$$)|(\$[ ]*\d+(\.\d+)?$)",
        ]
        self.reg_exs_substitutions = [
            "+12345678901",
            "x@example.com",
            "https://www.example.com",
            "ID",
            "Money",
        ]
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        ref = (
            importlib.resources.files("symspellpy")
            / "frequency_dictionary_en_82_765.txt"
        )
        with importlib.resources.as_file(ref) as dictionary_path:
            self.sym_spell.load_dictionary(dictionary_path, 0, 1)

        self.keywords = set()
        with open("data/PII_keywords.txt", "r") as file:
            for line in file:
                self.keywords.add(line.strip().lower())

    # Returns only the privacy relevant detections
    def filter(self, elements: Dict[int, Element], classification_file, tables={}):

        texts = {
            id: element
            for id, element in elements.items()
            if element.category == "Text"
        }
        text_fields = {
            id: element
            for id, element in elements.items()
            if element.category == "EditText"
        }

        # only check those elements that contain text
        private_ids = set()
        for id, text_element in texts.items():
            if text_element.text_content is not None:
                # Apply regex for phone numbers and mail adresses
                matched_regex = next(
                    (
                        (index, regex)
                        for index, regex in enumerate(self.reg_exs)
                        if re.match(regex, text_element.text_content)
                    ),
                    None,
                )
                if matched_regex is not None:
                    private_ids.add(id)
                    elements[id].privacy_type = "Regex"
                    elements[id].substitution = self.reg_exs_substitutions[
                        matched_regex[0]
                    ]

        # If in keyword list: find nearest component, redact its text:

        # Find nearest words
        refined_texts = self.refine_ocr_text(texts)

        allowed_characters = r"[^a-z0-9 .-]"
        for id, text_element in refined_texts.items():

            matched_keyword = next(
                (
                    keyword
                    for keyword in self.keywords
                    if keyword in text_element.text_content.lower()
                ),
                None,
            )
            if matched_keyword is not None:
                if not (
                    text_element.parent_id is not None
                    and elements[text_element.parent_id].category == "EditText"
                ):  # label can not be in editText itself
                    id_closest = self.find_closest_element(text_element, text_fields)
                    # get the children of the closest text field and redact it
                    if id_closest != -1 and text_fields[id_closest].children != []:
                        for child_id in text_fields[id_closest].children:
                            if elements[child_id].category == "Text":
                                # Store as private and reason
                                private_ids.add(child_id)
                                elements[child_id].privacy_type = "Keyword"
                                elements[child_id].substitution = matched_keyword

                # check if text might be name of table column
                if len(tables) > 0:
                    private_element_ids = self.filter_table_column(
                        text_element, texts, tables
                    )
                    for id in private_element_ids:
                        private_ids.add(id)
                        elements[id].privacy_type = f"Column: {matched_keyword}"
                        elements[id].substitution = matched_keyword  # column type

        # change later
        private_elements = [elements[id] for id in private_ids]
        
        # dump to json
        for element in elements.values():
            if element.category == "Text":
                text = {
                    "id": element.id,
                    "x1": element.x1,
                    "y1": element.y1,
                    "x2": element.x2,                            
                    "y2": element.y2,
                    "text": element.text_content,
                    "context" : "UNAVAILABLE",
                    "classification": (element.id in private_ids),
                    "reason": element.privacy_type,
                    "substitution": element.substitution
                }
                with open(classification_file, "r+") as file:
                    try:
                        info_obj = json.load(file)
                    except (json.JSONDecodeError):
                        info_obj = {"texts": []}
                    info_obj["texts"].append(text)
                    file.seek(0)
                    json.dump(info_obj, file, indent=4, ensure_ascii=False)
        return private_elements

    # returns closest component, -1 if no closest component exists
    def find_closest_element(
        self, element: Element, elements: Dict[int, Element], threshold=75
    ):
        closest = -1
        min_dist = 9999999
        for id, compo in elements.items():
            dist = distance(element, compo)
            if dist < min_dist and dist > 0 and dist < threshold:
                closest = id
                min_dist = dist
        return closest

    def refine_ocr_text(
        self, texts: Dict[int, Element], max_edit_distance=2
    ) -> Dict[int, Element]:
        refined_texts = texts.copy()

        for id, text in texts.items():
            if text.text_content != "" and text.text_content.lower() not in self.keywords:
                nearest_text = self.sym_spell.lookup(
                    text.text_content.lower(),
                    Verbosity.TOP,
                    max_edit_distance=max_edit_distance,
                    transfer_casing=True,
                    include_unknown=True,
                )[0]
                text_refined = nearest_text.term

                if (
                    nearest_text.distance <= max_edit_distance
                    and nearest_text.distance > 0
                ):  # something changed
                    refined_texts[id] = copy.deepcopy(texts[id])
                    refined_texts[id].text_content = text_refined
                    """print(
                        f"Changed '{texts[id].text_content}' to '{refined_texts[id].text_content}'"
                    )"""

        return refined_texts

    def filter_table_column(
        self, element: Element, elements: Dict[int, Element], tables: Dict
    ):
        private_element_ids = set()
        if element.cell != None:
            row = element.cell[0]
            column = element.cell[1]


            for table in tables.values():
                if row == 0:
                    for other_elements in table["df"][column]:
                        for other_element in other_elements:
                            if other_element.category == "Text" and other_element.id != element.id:
                                private_element_ids.add(other_element.id)

        return private_element_ids
