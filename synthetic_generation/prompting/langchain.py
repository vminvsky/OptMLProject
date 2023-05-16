from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import SimpleSequentialChain, SequentialChain

from dotenv import load_dotenv


load_dotenv()


class DataTemplates:
    """Class for storing the templates for the different generation tasks."""

    def sentiment_description_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help define certain sentiments: positive, neutral, negative.
        """,
            )
        )

        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""List three factors / charactersitics of text that can make it neutral, positive, and negative sentiment. Make sure to list three factors for each of the sentiments. 
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def sentiment_rewrite_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""
        You are an advanced AI writer. Your job is to help write examples of social media comments that conveys certain sentiments: positive, neutral, negative.
        """,
            )
        )
        
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["characteristics","text"],
                template="""A text can be positive, neutral, or negative. Here are the characteristics of these texts: {characteristics}. Rewrite the following tweet 9 times, 3 for each sentiment type. Make sure to align with the rules listed above. Make as few changes as possible to the underlying text without changing too many words:
{text}
Reply with the following structure:
Neutral
1.
2.
3.
Positive
1.
2.
3.
Negative
1.
2.
3.
""",
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])
