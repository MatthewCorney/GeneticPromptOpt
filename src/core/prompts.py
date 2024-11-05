prompt_templates = {
    "mutation":
        {
            "roles":
                [
                    "You are a program that when given a prompt intended for another LLM will edit it based on a given "
                    "consideration, while still preserving the meaning of the query. The aim is to create a new"
                    "prompt which will perform better at a downstream task than the original prompt",
                ],
            'prompts':
                [
                    "Re-Edit the prompt trying to make it more detailed",
                    "Re-Edit the prompt trying to make it less detailed",
                    "Re-Edit the prompt trying to make it more formal",
                    "Re-Edit the prompt trying to make it less formal",
                    "Re-Edit the prompt by adding in a facet about thinking through the problem",
                    "Re-Edit the prompt by adding in a facet about taking time to think about the response",
                    "Re-Edit the prompt the specified, if applicable, to make the response shorter",
                    "Re-Edit the prompt the specified, if applicable, to make the response longer",
                    "Re-Edit the prompt the specified, if applicable, to make the response more detailed",
                    "Re-Edit the prompt the specified, if applicable, to ask for the introduction of a relevant expert",
                    "Re-Edit the prompt the specified, if applicable, to make the response more creative",
                    "Re-Edit the prompt the specified, if applicable, to make the response less creative",
                    "Re-Edit the prompt the specified, if applicable, to make the response for creative",
                ]
        },
    "crossover":
        {
            "roles":
                ["You are a program which will be provided with two prompts, based on a given consideration combine "
                 "them into a new prompt whilst still preserving the meaning of the prompt. The aim is to create a new"
                 "prompt which will perform better at a downstream task than the parent prompts",
                 ],
            'prompts':
                [
                    "Combine the following two queries into a new prompt",
                    "Combine the following two queries into a new prompt, try to take more from the first prompt",
                    "Combine the following two queries into a new prompt, try to take more from the second prompt",
                    "Combine the following two queries into a new prompt, be creative",
                    "Combine the following two queries into a new prompt, don't be creative",

                ]
        }
}
