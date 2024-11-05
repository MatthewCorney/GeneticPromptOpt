from abc import ABC, abstractmethod
import random
import math


class SelectionFunction(ABC):

    @abstractmethod
    def select(self, *args, **kwargs) -> list[str]:
        """
        This is an abstract method that must be implemented by any subclass.
        It can take any number of arguments (as indicated by *args and **kwargs).
        """
        pass


class TopCandidates(SelectionFunction):
    def __init__(self, num_to_select: int = 10):
        """
        top n candidate selection

        :param num_to_select:
        """
        self.num_to_select = num_to_select

    def select(self, population, fitness_scores, ):
        """
        Returns the top n individuals from the population

        :param population:
        :param fitness_scores:
        :return:
        """
        paired = zip(population, fitness_scores)
        top_pairs = sorted(paired, key=lambda pair: pair[1], reverse=True)[:self.num_to_select]
        top_individuals = [pair[0] for pair in top_pairs]
        return top_individuals


class RouletteWheelSelection(SelectionFunction):
    def __init__(self, num_to_select: int = 10):
        """
        Uses roulette selection to build a new population

        :param num_to_select:
        """
        self.num_to_select = num_to_select

    def select(self, population, fitness_scores):
        """

        :param population:
        :param fitness_scores:
        :return:
        """
        total_fitness = sum(fitness_scores)
        selection_probs = [fitness / total_fitness for fitness in fitness_scores]
        selected_individuals = random.choices(population, weights=selection_probs, k=self.num_to_select)
        return selected_individuals


class TournamentSelection(SelectionFunction):
    def __init__(self, num_to_select: int = 10, tournament_size: int = 2):
        """

        :param num_to_select:
        :param tournament_size:
        """
        self.tournament_size = tournament_size
        self.num_to_select = num_to_select

    def select(self, population, fitness_scores):
        """

        :param population:
        :param fitness_scores:
        :return:
        """
        selected_individuals = []
        for _ in range(self.num_to_select):
            tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
            best_individual = max(tournament, key=lambda pair: pair[1])[0]
            selected_individuals.append(best_individual)
        return selected_individuals


class BoltzmannSelection(SelectionFunction):
    def __init__(self, num_to_select: int = 10, temperature: float = 10.0):
        """

        :param num_to_select:
        :param temperature:
        """
        self.temperature = temperature
        self.num_to_select = num_to_select
        self.initial_temperature = temperature
        self.decay_rate = 0.95

    def select(self, population, fitness_scores):
        """

        :param population:
        :param fitness_scores:
        :return:
        """
        boltzmann_fitness = [math.exp(fitness / self.temperature) for fitness in fitness_scores]
        total_boltzmann_fitness = sum(boltzmann_fitness)
        selection_probs = [bf / total_boltzmann_fitness for bf in boltzmann_fitness]
        selected_individuals = random.choices(population, weights=selection_probs, k=self.num_to_select)
        return selected_individuals

    def update_temperature(self, generation):
        """

        :param generation:
        :return:
        """
        self.temperature = self.initial_temperature * self.decay_rate ** generation
