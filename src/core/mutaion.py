import random
from abc import ABC, abstractmethod

from src.settings import client, settings
from src.input import NewPrompt
import json

# Load JSON data from a file
with open('src\\core\\prompts.json', 'r') as file:
    prompt_templates = json.load(file)


class PermutationFunction(ABC):

    @abstractmethod
    def permute(self, *args, **kwargs) -> str:
        """
        Abstract method that permutes a prompt
        :param args:
        :param kwargs:
        :return:
        """
        pass


class Mutation(PermutationFunction):
    def __init__(self):
        """
        Mutates a single prompt to give a new prompt
        """
        pass

    def permute(self, base_query: str, *args, **kwargs) -> str:
        """
        Chooses a random mutator prompt to mutate a prompt with
        :param base_query:
        :param args:
        :param kwargs:
        :return:
        """
        mutator_role = random.choice(prompt_templates["mutation"]['roles'])
        mutator_prompt = random.choice(prompt_templates["mutation"]['prompts'])
        completion = client.beta.chat.completions.parse(
            model=settings.model,
            messages=[
                {"role": "system", "content": mutator_role},
                {"role": "user", "content": f"{mutator_prompt}"
                                            f"original_prompt : {base_query}"},
            ],
            response_format=NewPrompt,
        )

        event = completion.choices[0].message.parsed.new_prompt
        return event


class Crossover(PermutationFunction):
    def __init__(self):
        """
        Combines two queries together to form a new prompt
        """
        pass

    def permute(self, prompt_1: str, prompt_2: str, *args, **kwargs) -> str:
        """
        Chooses a random crossover prompt to mutate a prompt with

        :param prompt_1:
        :param prompt_2:
        :param args:
        :param kwargs:
        :return:
        """
        mutator_role = random.choice(prompt_templates["crossover"]['roles'])
        mutator_prompt = random.choice(prompt_templates["crossover"]['prompts'])
        completion = client.beta.chat.completions.parse(
            model=settings.model,
            messages=[
                {"role": "system", "content": mutator_role},
                {"role": "user", "content": f"{mutator_prompt}"
                                            f"original_prompt_1 : {prompt_1}"
                                            f"original_prompt_2 : {prompt_2}"},
            ],
            response_format=NewPrompt,
        )

        event = completion.choices[0].message.parsed.new_prompt
        return event
