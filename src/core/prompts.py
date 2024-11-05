prompt_templates = {
    "mutation":
        {
            "roles":
                [
                    "You are part of Genetic-Optimization Algorithm whose objective is mutate a prompt to ensure randomess "
                    "within the given prompt yet it should be a direct derivative of the original. "
                    "Observe the problem description and make modifications to the original prompt.",

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
                ["You are part of Genetic-Optimization Algorithm whose objective is to create a child based on two "
                 "prompts, specifically a Control Prompt and an Additive Prompt. Such that, a new prompt is created "
                 "taking the Control Prompt as template to insert segments/important vehicles of the Additive Prompt."
                 " Ensure it still satisfies the problem description. Also, ensure the prompt is more detailed, "
                 "it definitely underperformed which requires us to optimize further, there should be stronger "
                 "directions and more complex instructions.",
                 ],
            'prompts':
                [
                    "Re-Edit the template/control prompt to create a child prompt which is inspired by the additive "
                    "prompt. It cannot be the same.",
                ]
        }
}
