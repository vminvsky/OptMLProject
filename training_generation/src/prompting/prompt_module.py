from prompting.prompts_langchain import DataTemplates
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import random


class PromptModule:
    """
    Class for handling different types of prompts in language chain.
    """

    def __init__(
            self,
            cfg,
            text,
            **kwargs
    ):
        """
        Initialize the PromptModule.

        Parameters:
        - task (str): The task for the module, must be either 'sarcasm' or 'sentiment'.
        - simple (bool, optional): A flag to determine if the module is simple or not.
        - grounding (bool, optional): A flag to determine if the module uses grounding or not.
        - taxonomy_generation (bool, optional): A flag to determine if the module generates taxonomy or not.
        - refinement (bool, optional): A flag to determine if the module is refinement based or not.

        Raises:
        - ValueError: If the simple flag is true, taxonomy_generation and refinement must both be false.
        - ValueError: If the task is not either 'sarcasm' or 'sentiment'.
        """
        self.task = cfg.experiment.task
        self.classify = cfg.experiment.classify
        self.text = text
        params = cfg.experiment.prompt_style

        self.prompt_langchain = dict(cfg.experiment.prompt_langchain)  # get the terms for decoding
        self.prompt_langchain["direction"] = kwargs.get("direction")

        self.params = params
        self.open_ai_keys = cfg.local.OPENAI_API_KEY
        sample_key = random.choice(self.open_ai_keys)

        self.simple = params.simple
        self.taxonomy_generation = params.taxonomy_generation
        self.refinement = params.refinement
        self.grounding = params.grounding

        self.generation_params = cfg.experiment.models.generation_parameters
        self.model_name = cfg.experiment.models.model_name

        self.validate_parameters(self.task, self.simple, self.grounding, self.taxonomy_generation, self.refinement)
        self.llm = ChatOpenAI(model_name=self.model_name, openai_api_key=sample_key,
                              **cfg.experiment.models.generation_parameters)

        self.prompts = self.extract_prompt(
            self.task,
            self.simple,
            self.grounding,
            self.taxonomy_generation,
            self.refinement,
            self.prompt_langchain["direction"],          # in case of taxonomy generation,
            cfg
        )

    def generate(self):
        n = len(self.prompts)

        if self.simple:
            prompt = ChatPromptTemplate.from_messages(self.prompts[0])
            chain = LLMChain(prompt=prompt, llm=self.llm)
            generated = chain.run(**self.prompt_langchain)
            return generated

        elif self.classify:
            prompt = ChatPromptTemplate.from_messages(self.prompts[0])
            chain = LLMChain(prompt=prompt, llm=self.llm)
            generated = chain.run(**{"text": self.text})
            return generated

        elif self.grounding:
            prompt = ChatPromptTemplate.from_messages(self.prompts[0])
            chain = LLMChain(prompt=prompt, llm=self.llm)
            generated = chain.run(**{"text": self.text}, **self.prompt_langchain)
            return generated

        if self.taxonomy_generation & (not self.refinement):
            prompt = ChatPromptTemplate.from_messages(self.prompts[0])
            taxonomy = self.prompts[1]
            chain = LLMChain(prompt=prompt, llm=self.llm)
            generated = chain.run(**{"text": self.text, "taxonomy": taxonomy}, **self.prompt_langchain)
            return generated

    @staticmethod
    def extract_prompt(task, simple, grounding, taxonomy_generation, refinement, direction, cfg):
        if task == "sarcasm":
            if simple:
                return [DataTemplates().sarcasm_simple_prompt()]

            elif taxonomy_generation:
                grounding_prompt = DataTemplates().sarcasm_taxonomy_generation()
                taxonomy = "\n".join(DataTemplates().sarcasm_sample_taxonomies(direction))
                if refinement:
                    refinement = DataTemplates().sarcasm_with_refinement()
                    return [grounding_prompt, taxonomy, refinement]
                else:
                    return [grounding_prompt, taxonomy]
            elif grounding:
                if cfg.experiment.rewrite:
                    return [DataTemplates().sarcasm_grounded_prompt()]
                else:
                    return [DataTemplates().sarcasm_grounded_no_rewrite_prompt()]
            if cfg.experiment.classify:
                return [DataTemplates().sarcasm_annotate_prompt()]
        elif task == "sentiment":
            if simple:
                return [DataTemplates().sentiment_simple_prompt()]
            elif grounding:
                return [DataTemplates().sentiment_grounded_prompt()]
            elif taxonomy_generation:
                grounding_prompt = DataTemplates().sentiment_taxonomy_generation()
                taxonomy = "\n".join(DataTemplates().sentiment_sample_taxonomies(direction))
                if refinement:
                    refinement = DataTemplates().sarcasm_with_refinement()
                    return [grounding_prompt, taxonomy, refinement]
                else:
                    return [grounding_prompt, taxonomy]
        else:
            raise NotImplementedError("Only sarcasm is implemented.")

    @staticmethod
    def validate_parameters(task, simple, grounding, taxonomy_generation, refinement):
        """
        Validate the parameters passed to the constructor.

        Parameters:
        - task (str): The task for the module.
        - simple (bool): The simple flag.
        - taxonomy_generation (bool): The taxonomy_generation flag.
        - refinement (bool): The refinement flag.

        Raises:
        - ValueError: If the simple flag is true, taxonomy_generation and refinement must both be false.
        - ValueError: If the task is not either 'sarcasm' or 'sentiment'.
        """
        # 'task' should be either 'sarcasm' or 'sentiment'
        if task not in ["sarcasm", "sentiment"]:
            raise NotImplementedError("Task must be either 'sarcasm' or 'sentiment'")

        #if not (simple or grounding or taxonomy_generation or refinement):
        #    raise ValueError("At least simple, taxonomy generation, grounding or refinement need to be True")

        # If 'simple' is true, both 'taxonomy_generation' and 'refinement' should be false
        if simple and (taxonomy_generation or refinement):
            raise ValueError(f"""For simple prompts, both taxonomy generation and refinement 
                             should be set to False. Now they are: {taxonomy_generation}, {refinement}""")

        if refinement and not taxonomy_generation:
            raise ValueError("Refinement only works with taxonomy generation.")
