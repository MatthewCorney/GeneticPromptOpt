# Introduction
Package for using genetic optimisation for prompt optimisation.
The general idea is with a small set of training data and some basic prompts for combing and mutating
the base prompt a better prompt can be generated.
The intention in the code is to provide a generalised workflow which can be extended with more detailed
and different evaluation/mutation strategies 

todo: add logging
todo: compare with other methods
todo: edit the 
# Dependencies
The package has the following dependencies:
Currently the pipeline only works for openai/chatgpt however this should be able to be extended with other openai
compliant apis

- numpy
- openai
- Levenshtein
- pydantic

# Usage

```
pip install git+https://github.com/MatthewCorney/GeneticPromptOpt.git
```
## Define our base classes for the input and output
Definging the outptut format is important so that openai will try to return data in the correct format
```
from pydantic import BaseModel, Field

from GeneticPromptOpt.core.evaluation import SimilarResponse
from GeneticPromptOpt.core.mutaion import Mutation, Crossover
from GeneticPromptOpt.core.population import MutationPopulationGeneration
from GeneticPromptOpt.core.selection import TopCandidates
from GeneticPromptOpt.genetic_prompt_opt import genetic_algorithm
from GeneticPromptOpt.input import BaseInputClass

class QuestionAnswerOutput(BaseModel):
    answer: str


class QuestionAnswerInput(BaseInputClass):
    question: str = Field(..., input=True)
    answer: str = Field(..., output=True)

    @property
    def response_model(self) -> type[BaseModel]:
        return QuestionAnswerOutput

```
### Load our data into the training class
The intention is for trickier problems with a small amount of gold standard responses, currencting
each prompt generated looks over all the training data so to keep cost down a small number of training
examples is expected
```
with open("data\\hotpot_train_v1.1.json",
          'r') as f:
    data_set = json.load(f)
training = []
for a in data_set[:10]:
    training.append(QuestionAnswerInput(question=a['question'],

                                        answer=a['answer']))

```
## Run the pipeline
All of these classes can easily be changed of extended
```
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

```