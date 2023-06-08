from dataclasses import dataclass, field
import os

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
import langchain.agents


with open("../API_KEY", "r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()


llm = ChatOpenAI(temperature=0.0)


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "text_prompt_input",
        "existing_code_input",
        "error_messages_input",
    ],
    template="""You are a python coding assistant and will be using the following information to help a user with their code related question.
                             
The existing code is:
-----
{existing_code_input}
-----

The error messages are:
-----
{error_messages_input}
-----

The user input is:
-----
{text_prompt_input}
-----
                                 
In your answer you should always follow these rules:
- Give your answer with Markdown formatting (especially for code blocks)
- Use the latest python programming techniques and best practices
- Use the latest python libraries when appropriate
- Address any error messages that are present
- Use the existing code as a starting point if present
- Include type hints where appropriate
- If the answer is a simple piece of code, only include the code and not the explanation

Based on the information above, give a response that will satisfy the user's request.
""",
)


@dataclass
class DisplayResponse:
    code: str
    summaries: str
    thought_process: str


def generate_response(text_prompt_input, existing_code_input, error_messages_input):
    chain = LLMChain(llm=llm, prompt=PROMPT_TEMPLATE)
    response = chain.run(
        {
            "text_prompt_input": text_prompt_input,
            "existing_code_input": existing_code_input,
            "error_messages_input": error_messages_input,
        }
    )
    return DisplayResponse(code=response, summaries="", thought_process="")
