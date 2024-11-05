import json

from pydantic import BaseModel, Field

from src.core.evaluation import SimilarResponse
from src.core.mutaion import Mutation, Crossover
from src.core.population import MutationPopulationGeneration
from src.core.selection import TopCandidates
from src.genetic_prompt_opt import genetic_algorithm
from src.input import BaseInputClass
import requests
import os


def download_hotpot_qa():

    # URL of the file to download
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"

    # Directory to save the file
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Path to save the downloaded file
    file_path = os.path.join(data_dir, "hotpot_train_v1.1.json")

    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    # Save the file to the specified path
    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"File downloaded and saved to {file_path}")

class QuestionAnswerOutput(BaseModel):
    answer: str


class QuestionAnswerInput(BaseInputClass):
    question: str = Field(..., input=True)
    answer: str = Field(..., output=True)

    @property
    def response_model(self) -> type[BaseModel]:
        return QuestionAnswerOutput


if __name__ == "__main__":

    download_hotpot_qa()

    with open("data\\hotpot_train_v1.1.json",
              'r') as f:
        data_set = json.load(f)
    training = []
    for a in data_set[:10]:
        training.append(QuestionAnswerInput(question=a['question'],

                                            answer=a['answer']))
    base_query = 'Answer the question provided'
    fitness_method = SimilarResponse()
    selection_method = TopCandidates()
    mutation_method = Mutation()
    crossover_method = Crossover()
    generate_population = MutationPopulationGeneration()
    result = genetic_algorithm(base_query=base_query,
                               training=training,
                               fitness_method=fitness_method,
                               selection_method=selection_method,
                               mutation_method=mutation_method,
                               crossover_method=crossover_method,
                               generate_population=generate_population,
                               population_size=5,
                               num_generations=3,
                               crossover_rate=0.50,
                               mutation_rate=0.50,
                               elitism_rate=0.5
                               )
    print(result['best_score'])
    print(result['best_individual'])
