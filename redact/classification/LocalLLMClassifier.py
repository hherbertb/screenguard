import instructor

from time import sleep
from typing import Dict, List
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

from activitygen.merge.element import Element


class PrivacyClassification(BaseModel):
    label: bool
    chain_of_thought: str = Field(
        description="Explanation for the classification of the potientially private term.",
    )


class PrivacyClassificationSubstitution(BaseModel):
    label: bool
    chain_of_thought: str
    substitution: str


class LocalLLMClassifier:
    def __init__(self, model_name="llama3.2", subsitute: bool = True):
        self.substitute = subsitute
        self.model = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        self.model_name = model_name

    # Returns only the privacy relevant detections
    def filter(
        self,
        elements: Dict[int, Element],
        prompt_type="ShortZeroShot",
        use_context=True,
    ):
        # Set target data model
        if self.substitute:
            response_model = PrivacyClassificationSubstitution
        else:
            response_model = PrivacyClassification
        private_ids = []
        path = f"configs/prompts/{prompt_type}.txt"

        texts = {
            id: element
            for id, element in elements.items()
            if element.category == "Text"
        }

        with open(path, "r") as prompt_file:
            base_prompt = prompt_file.read()

            for text in texts.values():
                prompt = base_prompt
                prompt += "Term: " + text.text_content

                if use_context:
                    prompt += ", Surrounding words: "
                    if len(text.context) == 0:
                        prompt += '"NONE"'
                    for context in text.context:
                        if context is not None:
                            prompt += (
                                '"' + texts[context.element_id].text_content + '", '
                            )
                    # Remove last ,
                    prompt = prompt[:-1]

                print(prompt)
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    response_model=response_model,
                    max_retries=6,
                )

                classification = response.label
                chain_of_thought = response.chain_of_thought
                if self.substitute:
                    substitution = response.substitution
                print(
                    f"{text.text_content}; {classification} ; {chain_of_thought}; {substitution}\n-----"
                )

                if classification:
                    private_ids.append(text.id)
                    texts[text.id].privacy_type = "llm"
                    if self.substitute:
                        texts[text.id].substitution = substitution
                    else:
                        texts[text.id].substitution = "REDACTED"

        private_elements = [elements[id] for id in private_ids]
        return private_elements
