from abc import ABC, abstractmethod

from src.core.mutaion import Mutation


class PopulationGeneration(ABC):

    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Abstract method that generates an initial population
        :param args:
        :param kwargs:
        :return:
        """
        pass


class MutationPopulationGeneration(PopulationGeneration):
    def __init__(self):
        pass

    def generate(self, base_query: str, population_size: int = 10) -> list[str]:
        """
        Takes a base query and generates an initial population of n prompts

        :param base_query:
        :param population_size:
        :return:
        """
        prompts = []
        for i in range(population_size):
            mutator = Mutation()
            mutated_response = mutator.permute(base_query)
            prompts.append(mutated_response)
        return prompts
