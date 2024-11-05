from abc import ABC, abstractmethod

import Levenshtein
import numpy as np

from src.input import BaseInputClass
from typing import List
from src.settings import client, settings, logger


class FitnessFunction(ABC):

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float:
        """
        Abstract method that runs an evaluation over a batch
        :param args:
        :param kwargs:
        :return:
        """
        pass


class ExactResponse(FitnessFunction):

    def evaluate(self, predicted_responses: str, gold_standard: List[BaseInputClass]) -> float:
        """
        Looks to see if the response is exactly the same as the gold standard responses
        :param predicted_responses:
        :param gold_standard:
        :return:
        """
        logger.debug("Running evaluation")
        running_score = []
        for predicted, gold in zip(predicted_responses, gold_standard):
            values = []
            for k, v in gold.get_output_fields().items():
                if predicted[k] == v:
                    values.append(1)
                else:
                    values.append(0)
            running_score.append(np.mean(values))
        return float(np.mean(running_score))


class SimilarResponse(FitnessFunction):

    def evaluate(self, predicted_responses: List[str], gold_standard: List[BaseInputClass]) -> float:
        """
        looks to see if the response is similar to the gold standard response (by string based similarity)
        :param predicted_responses:
        :param gold_standard:
        :return:
        """
        logger.debug("Running evaluation")
        running_score = []
        for predicted, gold in zip(predicted_responses, gold_standard):
            values = []
            for k, v in gold.get_output_fields().items():
                values.append(Levenshtein.ratio(predicted[k], v))
            running_score.append(np.mean(values))
        return float(np.mean(running_score))


class SemanticallySimilarResponse(FitnessFunction):
    def __init__(self):
        """
        Uses vector embeddings from a llm to compare the responses
        """

    def evaluate(self, predicted_responses: List[str], gold_standard: List[BaseInputClass]) -> float:
        """
        looks to see if the response is similar to the gold standard response (by sematic based similarity)

        :param predicted_responses:
        :param gold_standard:
        :return:
        """
        logger.debug("Running evaluation")
        running_score = []
        for predicted, gold in zip(predicted_responses, gold_standard):
            values = []
            for k, v in gold.get_output_fields().items():
                predicted_vector = client.embeddings.create(input=[predicted[k]], model=settings.embedding_model).data[
                    0].embedding
                gold_vector = client.embeddings.create(input=[v], model=settings.embedding_model).data[0].embedding
                cosine_similarity = np.dot(predicted_vector, gold_vector) / (
                        np.linalg.norm(predicted_vector) * np.linalg.norm(gold_vector))
                values.append(cosine_similarity)
            running_score.append(np.mean(values))
        return float(np.mean(running_score))
