import random

from src.core.evaluation import FitnessFunction, SimilarResponse
from src.input import BaseInputClass
from src.core.mutaion import PermutationFunction, Mutation, Crossover
from src.settings import client
from src.settings import settings, logger

from src.core.population import PopulationGeneration, MutationPopulationGeneration
from src.core.selection import SelectionFunction, TopCandidates


def predict(prompt: str, minibatch: list[BaseInputClass]) -> list[dict]:
    """

    :param prompt:
    :param minibatch:
    :return:
    """
    predicted_responses = []
    for i in minibatch:
        try:
            completion = client.beta.chat.completions.parse(
                model=settings.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": i.form_input_query()},
                ],
                response_format=i.response_model,
            )
            event = completion.choices[0].message.parsed
            outputs = i.get_output_fields().keys()
            results = {key: getattr(event, key) for key in outputs}
            predicted_responses.append(results)
        except Exception as e:
            logger.error(e)
            logger.error(f"Issue with prediction for instance {i}"
                         f"With prompt {prompt}"
                         )
    return predicted_responses


def genetic_algorithm(base_query: str,
                      training: list[BaseInputClass],
                      fitness_method: FitnessFunction = SimilarResponse,
                      selection_method: SelectionFunction = TopCandidates,
                      mutation_method: PermutationFunction = Mutation,
                      crossover_method: PermutationFunction = Crossover,
                      generate_population: PopulationGeneration = MutationPopulationGeneration,
                      population_size: int = 20,
                      num_generations: int = 10,
                      crossover_rate: float = 0.50,
                      mutation_rate: float = 0.50,
                      elitism_rate: float = 0.05,
                      ) -> dict:
    """
    Base function for running the optimisation

    :param base_query: Query to Optimise
    :param training: Training examples
    :param fitness_method: Initialised class to evaluate the predictions
    :param selection_method: Initialised class to decide how to select individuals based on fitness
    :param mutation_method: Initialised class to decide how to select mutate prompts
    :param crossover_method: Initialised class to decide how to combine prompts
    :param generate_population: Initialised class to decide how to generate an initial population
    :param population_size: Size of the population/generations
    :param num_generations: Number of generation to run for
    :param crossover_rate: Prompt combination rate
    :param mutation_rate: Prompt mutation rate
    :param elitism_rate: Generational carry over rate
    :return:
    """
    logger.info(f"Number of queries expected to be sent off "
                f"{(population_size * len(training) * num_generations) + (population_size * num_generations * (mutation_rate * crossover_rate))}")

    population = generate_population.generate(base_query, population_size)
    population.append(base_query)
    logger.info(f'Population generated {population}')
    fitness_scores = []
    logger.info(f'Calculating fitness of initial population')
    for prompt in population:
        predicted_responses = predict(prompt=prompt, minibatch=training)
        fitness_score = fitness_method.evaluate(predicted_responses=predicted_responses, gold_standard=training)
        fitness_scores.append(fitness_score)

    generations = []
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    best_fitness = max(fitness_scores)
    logger.info(f'Best Fitness {best_fitness}')
    logger.info(f'Best Individual {best_individual}')
    generations.append({'population': population, 'scores': fitness_scores, 'generation': 0})
    for generation in range(num_generations):
        logger.info(f'Running generation={generation}')
        best_individuals = selection_method.select(population, fitness_scores)

        logger.debug(f'Creating offspring based on {crossover_rate=}')
        offspring = []
        for _ in range(population_size):
            if random.random() > crossover_rate:
                parent1, parent2 = random.sample(best_individuals, 2)
                child = crossover_method.permute(prompt_1=parent1, prompt_2=parent2)
                offspring.append(child)
            else:
                best_individual = random.sample(best_individuals, 1)[0]
                offspring.append(best_individual)

        logger.debug(f'Mutating offspring based on {mutation_rate=}')
        mutated_children = []
        for child in offspring:
            if random.random() > mutation_rate:
                mutated_child = mutation_method.permute(child)
                mutated_children.append(mutated_child)
            else:
                mutated_children.append(child)

        logger.debug(f'Getting offspring fitness')
        offspring_fitness = []
        for prompt in mutated_children:
            predicted_responses = predict(prompt=prompt, minibatch=training)
            offspring_individual_fitness = fitness_method.evaluate(predicted_responses=predicted_responses,
                                                                   gold_standard=training
                                                                   )
            offspring_fitness.append(offspring_individual_fitness)
        sorted_offspring = sorted(zip(offspring, offspring_fitness), key=lambda x: x[1], reverse=False)

        num_elite = int(elitism_rate * population_size)
        offspring, offspring_fitness = zip(*sorted_offspring)
        offspring = list(offspring)
        offspring_fitness = list(offspring_fitness)
        logger.debug(f'Replacing worst candidates with elite individuals from {generation - 1}')
        if num_elite > 0:
            elite_individuals_zip = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[
                                    :num_elite]
            elite_individuals = [individual for individual, fitness in elite_individuals_zip]
            elite_fitness = [fitness for individual, fitness in elite_individuals_zip]
            offspring[:num_elite] = elite_individuals  # Replace the worst offspring with the best elite individuals
            offspring_fitness[:num_elite] = elite_fitness

        population = offspring
        fitness_scores = offspring_fitness
        current_best_individual = population[fitness_scores.index(max(fitness_scores))]
        current_best_fitness = max(fitness_scores)
        logger.info(f'Best Fitness in generation = {generation} fitness = {current_best_fitness}')
        logger.info(f'Best Individual generation = {generation} prompt ={current_best_individual}')
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
        generations.append({'population': population, 'scores': fitness_scores, 'generation': generation})

    # Return the best individual found
    return {'best_individual': best_individual,
            'best_score': max(fitness_scores),
            'generations': generations
            }
