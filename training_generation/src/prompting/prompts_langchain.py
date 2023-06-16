from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import numpy as np

from langchain.schema import HumanMessage, AIMessage, SystemMessage

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
                template="""List five factors of text that can make it neutral, positive, or negative in sentiment. 
                Make sure to list three factors for each of the sentiments.""",
            )
        )
        return [system_message, human_message]

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
                input_variables=["characteristics", "text"],
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
        return [system_message, human_message]

    def sarcasm_simple_prompt(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that generates sarcastic and non-sarcastic texts."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["num_generations", "direction"],
                template="Generate {num_generations} {direction} texts. Ensure diversity in the generated texts."
            )
        )

        return [system_message, human_message]


    def sarcasm_annotate_prompt(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that annotates sarcastic and non-sarcastic texts."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text"],
                template="""Classify the following text as being sarcastic or non-sarcastic. Reply with 'Sarcastic' if it's sarcastic and 'Non-sarcastic' if it's non-sarcastic. 
                Text: {text}
                """
            )
        )
        return [system_message, human_message]

    def sarcasm_grounded_prompt(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that generates sarcastic and non-sarcastic texts."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "num_generations", "direction"],
                template="""Rewrite the following text {num_generations} times to make it {direction}. 
                Make as few changes as possible to the text and stay true to its underlying style. 
                Text: {text}
                """
            )
        )
        return [system_message, human_message]

    def sarcasm_grounded_no_rewrite_prompt(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that generates sarcastic and non-sarcastic texts."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "num_generations", "direction"],
                template="""Here is an example of a {direction} text. Write {num_generations} new similar examples that have same {direction} tone. 
                Text: {text}
                """
            )
        )
        return [system_message, human_message]

    def sarcasm_taxonomy_creation(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that thinks about ways a text can be sarcastic and non-sarcastic texts."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["num_ideas", "direction"],
                template="""Come up with {num_ideas} ways a text can be {direction}.
                """
            )
        )
        return [system_message, human_message]

    def sarcasm_taxonomy_generation(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that generates sarcastic and non-sarcastic texts."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["taxonomy", "num_generations", "direction", "text"],
                template="""
                Here are the ways a text can be {direction}: 
                {taxonomy}
                
                Your task it to rewrite the following text {num_generations} times to make it {direction}.
                For each rewrite, select one of the ways and use it. 
                Make as few changes as possible to the text and stay true to its underlying style. 
                
                Text: {text}
                """
            )
        )
        return [system_message, human_message]

    def sarcasm_sample_taxonomies(self, direction) -> str:
        sarcastic_examples = [
            "Verbal Irony: Saying something but meaning the exact opposite.",
            "Socratic Sarcasm: Using rhetorical questions to deliver sarcasm.",
            "Self-Deprecating Sarcasm: Making sarcastic remarks about oneself.",
            "Hyperbolic Sarcasm: Overexaggerating to make a sarcastic point.",
            "Deadpan Sarcasm: Delivering a sarcastic remark with a serious or emotionless tone.",
            "Sarcastic Mimicry: Imitating or repeating others' statements sarcastically.",
            "Situational Sarcasm: Making a sarcastic comment in response to a specific situation.",
            "Prophecy Sarcasm: Making sarcastic predictions about future events.",
            "Benevolent Sarcasm: Using softer sarcasm that isn't meant to hurt the receiver.",
            "Cynical Sarcasm: Using sarcasm to harshly criticize or mock.",
            "Dramatic Sarcasm: Using exaggerated tones and gestures to deliver sarcasm.",
            "Polite Sarcasm: Using polite language or compliments to convey sarcasm.",
            "Manic Sarcasm: Using overly enthusiastic language to highlight a negative situation.",
            "Understated Sarcasm: Using understatement to highlight the absurdity of a situation.",
            "Sarcasm of Ignorance: Pretending to ignore or not understand the obvious.",
            "Ambiguous Sarcasm: Leaving the reader unsure whether the statement was sarcastic or not.",
            "Metaphoric Sarcasm: Using metaphors to convey sarcasm.",
            "Satirical Sarcasm: Exposing foolishness or corruption through sarcasm.",
            "Sardonic Sarcasm: Using harsh sarcasm aimed at a specific person or group.",
            "Poetic Sarcasm: Using a poetic or literary style to articulate sarcasm.",
            "Conditional Sarcasm: Expressing sarcasm by creating conditions for unlikely events.",
            "Euphemistic Sarcasm: Using mild or indirect expressions to convey sarcasm.",
            "Paradoxical Sarcasm: Using paradoxical statements to convey sarcasm.",
            "Misdirection Sarcasm: Using an unexpected twist at the end of a statement to convey sarcasm.",
            "Caps Lock Sarcasm: Using all caps in written communication for sarcastic emphasis.",
            "Innocent Sarcasm: Feigning naivety while making a sarcastic remark.",
            "Reversal Sarcasm: Speaking the opposite of what is expected in a sarcastic tone.",
            "Critical Sarcasm: Delivering negative or critical observations in a sarcastic tone.",
            "Juxtaposition Sarcasm: Using contrasting ideas or situations together to create sarcasm.",
            "Silent Sarcasm: Using ellipsis or omissions in written communication to create sarcasm."
        ]
        non_sarcastic_examples = [
            "Directness: The text is straightforward and without any hidden meanings.",
            "Genuine compliments: If the text includes a compliment that seems genuine, it's likely not sarcastic.",
            "Clear communication: The text is clear and communicates its intent without ambiguity.",
            "Informative: The text is simply providing information without any form of mockery or sarcasm.",
            "Empathetic statements: Expressions of understanding or empathy are typically sincere and not sarcastic.",
            "Expressing gratitude: If the text shows appreciation or gratitude, it's probably not sarcastic.",
            "Constructive criticism: The text provides clear and helpful feedback instead of being sarcastic.",
            "Positive tone: The text uses positive words and phrases that convey a positive attitude.",
            "Objective facts: The text provides objective facts without any opinion or irony.",
            "Personal expression: The text talks about personal experiences or emotions sincerely.",
            "Making Plans: A text that proposes future actions or plans.",
            "Affirmations: Genuine affirmations are typically non-sarcastic.",
            "Congratulatory Messages: Messages that genuinely congratulate someone for an achievement.",
            "Recommendations: Genuine suggestions or advice.",
            "Providing a Definition or Explanation: Straightforward explanations or definitions are typically non-sarcastic.",
            "Expressing Concern: Genuine expressions of worry or concern are usually non-sarcastic.",
            "Quotes or Proverbs: Quotes or proverbs shared for inspiration or wisdom are generally non-sarcastic.",
            "Relating Personal Experiences: Sharing personal experiences or anecdotes without any mocking undertones.",
            "Showing Interest: Genuine questions about the other person's life or experiences are usually non-sarcastic.",
            "Expression of Hope: Messages that express hope for the future are generally non-sarcastic.",
            "Fact-based: The text relies on concrete facts and figures.",
            "Apologies: A sincere apology is generally not sarcastic.",
            "Asking questions: Genuine inquiries and requests for clarification are usually non-sarcastic.",
            "Sharing news: Simply informing about an event or situation is typically non-sarcastic.",
            "Encouraging statements: Encouraging words and phrases can be genuine and uplifting, like 'Keep up the great work!'",
            "Expressing personal feelings: A text that openly expresses feelings is less likely to be sarcastic.",
            "Instructions or guidelines: Providing directions or steps to accomplish a task is typically straightforward.",
            "Invitations: Invitations are usually sincere and welcoming.",
            "Narrating a story: A text recounting a past event or telling a story is usually non-sarcastic."
        ]
        prompts = {"sarcastic": sarcastic_examples, "non-sarcastic": non_sarcastic_examples}
        random_ten = list(np.random.choice(prompts[direction], size=10))
        return random_ten

    def sarcasm_refinement(self) -> ChatPromptTemplate:
        raise NotImplementedError("Not implemented")

    def sentiment_simple_prompt(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that generates texts that possess a positive, negative, or neutral sentiment."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["num_generations", "direction"],
                template="Generate {num_generations} texts with a {direction} sentiment. Ensure diversity in the generated texts."
            )
        )

        return [system_message, human_message]


    def sentiment_grounded_prompt(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that generates texts that possess a positive, negative, or neutral sentiment."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["text", "num_generations", "direction"],
                template="""Rewrite the following text {num_generations} times to make it {direction}. 
                Make as few changes as possible to the text and stay true to its underlying style. 
                Text: {text}
                """
            )
        )
        return [system_message, human_message]

    def sentiment_taxonomy_generation(self) -> list:
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="You are a model that generates texts that possess a positive, negative, or neutral sentiment."
            )
        )
        human_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["taxonomy", "num_generations", "direction", "text"],
                template="""
                Here are the ways a text can be {direction}: 
                {taxonomy}

                Your task it to rewrite the following text {num_generations} times to make it {direction}.
                For each rewrite, select one of the ways and use it. 
                Make as few changes as possible to the text and stay true to its underlying style. 

                Text: {text}
                """
            )
        )
        return [system_message, human_message]

    def sentiment_sample_taxonomies(self, direction) -> str:
        positive_examples = [
            "Use of Positive Words and Phrases: Texts with positive sentiment often include words that evoke happiness, satisfaction, joy, excitement, and other pleasant emotions.",
            "Inclusion of Compliments: Positive sentiment can be portrayed through compliments and praises.",
            "Expressions of Gratitude: Texts that express gratitude typically convey a positive sentiment.",
            "Celebration of Achievements: Positivity can be expressed through acknowledgment and celebration of personal or collective achievements, successes, and milestones.",
            "Use of Positive Emojis: Emojis like smiley faces, hearts, thumbs up, and other symbols that represent happiness can create a positive sentiment in text communication.",
            "Use of Exclamation Marks: While excessive usage can be overwhelming, a well-placed exclamation mark can add positivity to a statement, showing enthusiasm or excitement.",
            "Highlighting Positive Aspects: Even when discussing something with negative aspects, focusing on the positive aspects or the silver linings can indicate positive sentiment.",
            "Offering Encouragement or Motivation: Texts that include words of encouragement or motivation often convey a positive sentiment.",
            "Sharing Good News: Positive sentiment is often associated with sharing good news or exciting updates.",
            "Use of Uplifting Quotes or Affirmations: Incorporating positive, inspirational quotes or affirmations can also give the text a positive sentiment.",
            "Positive Predictions and Expectations: Expressing optimism about the future or anticipation for upcoming events contributes to a positive sentiment.",
            "Words of Appreciation: Acknowledging someone's effort, time, or contribution increases the positivity of the text.",
            "Constructive Feedback: Even when providing feedback, focusing on the strengths, potential improvements, and using a supportive tone can give the text a positive sentiment.",
            "Use of Polite Language: Courteous phrases can make requests or commands sound more positive and less demanding.",
            "Sharing Personal Achievements or Success Stories: Sharing success or achievements, no matter how small, can add positivity to the text.",
            "Inclusion of Hopeful Statements: Expressions of hope contribute to a positive tone.",
            "Mentioning Positive Attributes or Qualities: Describing someone or something with positive adjectives can enhance positivity.",
            "Use of Jovial Tone: Using light-hearted, playful, or humorous language (when appropriate) can make a text feel more positive.",
            "Words of Support: Showing support or solidarity often signals positivity.",
            "Highlighting Progress or Improvement: Recognizing and appreciating progress, no matter how small, can be another way to imbue the text with positive sentiment.",
            "Use of Reassuring Statements: Using phrases that provide reassurance in a challenging situation can lend a positive tone to the text.",
            "Celebrating Small Wins: Even minor victories can imbue positivity in a text.",
            "Sharing Positive Experiences or Memories: Talking about positive past experiences or nostalgic memories can add a positive sentiment to the text.",
            "Offering Help or Assistance: Phrases that show willingness to support can make a text sound more positive.",
            "Positive Assumptions: Statements that show faith in the other person's capabilities add positivity.",
            "Invitations for Positive Interaction: Inviting someone to a pleasant event or activity can add a positive tone.",
            "Expressions of Admiration or Respect: Complimenting someone's abilities, actions, or qualities shows respect and admiration, thus adding a positive tone.",
            "Acknowledgment of Effort: Recognizing the effort someone has put into a task, even if the outcome was not perfect, can convey a positive sentiment.",
            "Use of Enthusiastic Language: Words that express enthusiasm can create a positive sentiment.",
            "Showing Understanding and Empathy: Empathetic statements can contribute to a positive sentiment, as they demonstrate care and understanding."
        ]
        negative_examples = [
            "Sarcasm: Text uses irony or mock praise to convey contempt or ridicule.",
            "Criticism: The text is largely judgmental, criticizing people or things.",
            "Insults: Offensive and rude remarks are made about a person or thing.",
            "Disrespect: The text belittles or disregards the rights, feelings, or opinions of others.",
            "Discontent: The text communicates dissatisfaction or unhappiness with someone or something.",
            "Stereotyping: The text generalizes negatively about a group of people.",
            "Pessimism: The text communicates a negative or hopeless outlook.",
            "Anger: The text conveys intense displeasure, outrage or wrath.",
            "Hate speech: The text expresses intense hostility or violence towards an individual or group.",
            "Cynicism: The text expresses a disbelief in the sincerity or goodness of human motives or actions.",
            "Blame: The text assigns fault or responsibility for a wrongdoing or mistake to someone.",
            "Threats: The text communicates intent to cause harm or loss to someone or something.",
            "Defamation: The text damages the good reputation of someone with false or unsubstantiated claims.",
            "Fear Mongering: The text incites fear or spreads terrifying rumors about something or someone.",
            "Negativity bias: The text focuses predominantly on negative aspects, events, or outcomes.",
            "Manipulation: The text attempts to influence someone in a deceitful or unfair way.",
            "Profanity: The text uses coarse language considered offensive or vulgar.",
            "Victim blaming: The text places blame on victims for the harmful events that have happened to them.",
            "Guilt-Tripping: The text uses guilt as a form of emotional manipulation to make someone feel responsible or at fault.",
            "Gaslighting: The text aims to make someone doubt their own perceptions or memories.",
            "Bullying: The text includes repeated, harmful actions towards an individual or group.",
            "Trolling: The text aims to provoke or upset people, often for the author's amusement.",
            "Dismissiveness: The text disregards or trivializes someone's feelings, thoughts, or experiences.",
            "Rudeness: The text displays a lack of respect or manners.",
            "Doomsaying: The text anticipates or predicts disaster or the worst possible outcome.",
            "Hostility: The text shows an unfriendly and aggressive behavior or attitude.",
            "Prejudice: The text shows preconceived negative opinions that are not based on reason or actual experience.",
            "Discrimination: The text treats people differently, particularly in a worse way, from others because of their race, religion, sex, etc.",
            "Passive Aggression: The text indirectly expresses negative feelings instead of openly addressing them.",
            "Shaming: The text uses criticism or mockery to make an individual feel embarrassed or inadequate."
        ]
        neutral_examples = [
            "Statement of Fact: Text communicates information that is objective and not influenced by personal feelings or opinions.",
            "Instructional Content: The text provides guidance or direction without expressing sentiment.",
            "News Reporting: The text presents an unbiased account of events or situations.",
            "Technical Writing: The text provides detailed information about a particular subject matter, typically related to science or technology.",
            "Description: The text provides an account of the features or qualities of something without any emotion or opinion.",
            "Presentation of Data: The text communicates statistics, facts or numerical information.",
            "Academic Writing: The text presents research findings or scholarly information.",
            "Neutral Questions: The text asks questions that don't lead the reader to a particular sentiment.",
            "Definitions: The text provides the meanings of words or terms.",
            "Summarization: The text condenses longer pieces of information, focusing on the key points without displaying any sentiment.",
            "Neutral Suggestions: The text proposes ideas or advice without any positive or negative connotations.",
            "Informative Content: The text provides useful information without expressing personal feelings.",
            "Neutral Reviews: The text evaluates something based on objective criteria without conveying personal sentiment.",
            "Procedural Explanation: The text describes how something is done or how to do something without any emotional context.",
            "Neutral Observations: The text records what has been seen or observed without expressing any sentiment.",
            "Transactional Language: The text communicates necessary information for completing a transaction.",
            "Historical Account: The text provides a record of past events without showing any personal bias.",
            "Scientific Findings: The text presents results from a scientific study without personal interpretation or sentiment.",
            "Neutral Comparison: The text compares two or more things without showing a preference.",
            "Neutral Forecast: The text predicts future occurrences based on data and facts, devoid of sentiment.",
            "Neutral Advice: The text offers guidance or recommendations without expressing personal feelings.",
            "Quotations: The text quotes someone's words verbatim without adding personal sentiment.",
            "Neutral Argument: The text presents a balanced view of different sides of an issue.",
            "Legal Language: The text uses language specific to laws and regulations, typically devoid of sentiment.",
            "Neutral Evaluation: The text assesses or judges something based on factual criteria.",
            "Neutral Narrative: The text tells a story or describes a sequence of events without incorporating personal sentiment.",
            "Neutral Opinion: The text states a point of view without strong emotional connotations.",
            "Neutral Reporting: The text reports an event or situation without any bias.",
            "Neutral Analysis: The text examines a topic in detail to explain it, without expressing personal sentiment.",
            "Neutral Recommendation: The text suggests a course of action without expressing strong feelings or bias."
        ]
        prompts = {"neutral": neutral_examples, "positive": positive_examples, "negative": negative_examples}
        random_ten = list(np.random.choice(prompts[direction], size=10))
        return random_ten
