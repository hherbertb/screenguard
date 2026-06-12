import json
import instructor
import math
import google.generativeai as genai


from instructor.exceptions import InstructorRetryException
from time import sleep
from typing import Dict, List
from configparser import ConfigParser
from pydantic import BaseModel

from activitygen.merge.element import Element


class PrivacyClassifications(BaseModel):
    reasons: List[str]
    labels: List[bool]


class PrivacyClassificationsSubstitution(BaseModel):
    reasons: List[str]
    labels: List[bool]
    substitutions: List[str]


class LLMPrivacyClassifier:
    def __init__(self, subsitute: bool = True):
        self.substitute = subsitute
        self.model_name = "gemini-2.0-flash"

        config_parser = ConfigParser()
        config_parser.read("configs/api_keys.cfg")
        api_key = config_parser.get("MAIN", "GEMINI_API_KEY")
        genai.configure(api_key=api_key)

        self.model = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name=self.model_name,
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )
        print(f"Using model {self.model_name}")

    # Returns only the privacy relevant detections
    def filter(
        self,
        elements: Dict[int, Element],
        classification_file,
        prompt_type="ShortFewShot2",
        bundle_size=5,
        use_context=True,
    ):
        # Set target data model
        if self.substitute:
            response_model = PrivacyClassificationsSubstitution
        else:
            response_model = PrivacyClassifications

        private_ids = []
        path = f"configs/prompts/{prompt_type}.txt"

        texts = {
            id: element
            for id, element in elements.items()
            if element.category == "Text"
        }

        sub_prompts = self.texts_to_llm_input(
            texts, bundle_size, use_context=use_context
        )

        with open(path, "r") as prompt_file:
            base_prompt = prompt_file.read()

        for i, sub_prompt in enumerate(sub_prompts):
            """if (i != 0) and (i % 15 == 0):  # wait for new api quota
                print("QUOTA EXHAUSTED: Wait 60 seconds...")
                sleep(61)
                print("Continue!")"""
            prompt = base_prompt
            prompt += sub_prompt

            noAnswer = True
            while noAnswer:
                try:
                    response = self.model.messages.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        response_model=response_model,
                    )
                    noAnswer = False
                except InstructorRetryException as e:
                    print("QUOTA EXHAUSTED: Wait 60 seconds...")
                    sleep(61)
                    print("Continue!")

            print(prompt)
            print(response.labels)
            print(response.reasons)

            classifications = response.labels # [True if x == "PRIVATE" else False for x in response.labels]
            reasons = response.reasons
            if self.substitute:
                substitutions = response.substitutions

            if len(classifications) < bundle_size:
                print(f"Not exactly {bundle_size} classifications returned.")
                print(f"For prompt: {prompt} we have {len(classifications)} classifications.")
            if self.substitute:
                print(substitutions)

            for j, cls in enumerate(classifications):
                if i * bundle_size + j < len(texts):
                    text_id = list(texts.values())[i * bundle_size + j].id
                    if cls == True:
                        private_ids.append(text_id)
                        texts[text_id].privacy_type = "llm"
                        if self.substitute:
                            texts[text_id].substitution = substitutions[j]
                        else:
                            texts[text_id].substitution = "REDACTED"

                    # dump to json
                    contexts = []
                    if use_context:
                        if len(texts[text_id].context) == 0:
                            contexts = ["NONE"]
                        for context in texts[text_id].context:
                            if context is not None:
                                contexts.append(texts[context.element_id].text_content)
                    else:
                        contexts = ["_DISABLED_"]
                    text = {
                        "id": text_id,
                        "x1": texts[text_id].x1,
                        "y1": texts[text_id].y1,
                        "x2": texts[text_id].x2,                            
                        "y2": texts[text_id].y2,
                        "text": texts[text_id].text_content,
                        "context" : contexts,
                        "classification": classifications[j],
                        "reason": reasons[j],
                        "substitution": substitutions[j]
                    }
                    with open(classification_file, "r+") as file:
                        try:
                            info_obj = json.load(file)
                        except (json.JSONDecodeError):
                            info_obj = {"texts": []}
                        info_obj["texts"].append(text)
                        file.seek(0)
                        json.dump(info_obj, file, indent=4, ensure_ascii=False)
                            

        private_elements = [elements[id] for id in private_ids]
        return private_elements

    # Converts text elements to string for llm input.
    # Generates len(texts) / bundle_size prompts
    def texts_to_llm_input(
        self, texts: Dict[int, Element], bundle_size, use_context=True
    ):
        bundled_prompts = []
        n_prompts = math.ceil(len(texts) / bundle_size)
        text_vals = texts.values()

        for i in range(n_prompts):
            prompt = ""
            for j in range(bundle_size):
                if i * bundle_size + j < len(text_vals):
                    element = list(text_vals)[i * bundle_size + j]
                    prompt += "Term: " + element.text_content

                    if use_context:
                        prompt += ", Surrounding words: "
                        if len(element.context) == 0:
                            prompt += '"NONE"'
                        for context in element.context:
                            if context is not None:
                                prompt += (
                                    '"' + texts[context.element_id].text_content + '", '
                                )
                        # Remove last ,
                        prompt = prompt[:-1]

                    prompt += "\n"

            bundled_prompts.append(prompt)

        return bundled_prompts
