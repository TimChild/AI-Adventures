import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from dash import dcc

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import langchain.agents
import os
import datetime

with open('../API_KEY', 'r') as f:
    os.environ['OPENAI_API_KEY'] = f.read()

HISTORY_FOLDER = 'chat_history'
os.makedirs(HISTORY_FOLDER, exist_ok=True)

embedding = OpenAIEmbeddings()
db = FAISS.load_local("faiss_dbs", embeddings=embedding, index_name="igor-test")

# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
# Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
#
# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:
# """)
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation history: 

---------------
CONVERSATION HISTORY:
{chat_history}
---------------

and follow up question:

---------------
FOLLOW UP QUESTION: 
{question}
---------------

Make a list of the information from the CONVERSATION HISTORY that is relevant to the FOLLOW UP QUESTION (condensing the information when possible), then rephrase the follow up question to be a standalone question. You should format your response like:
---------------
Possibly Relevant info:
- <relevant info 1>
- <relevant info 2>
- etc

<standalone question here>
---------------
""")

GENERATE_ANSWER_SYSTEM_PROMPT = """Use the following pieces of context that come from the "Igor Help Files" to answer the users question. 
-----------------------

{context}

-----------------------


When writing your answer to the users question you should follow these rules:
- If you don't know the answer, don't try to make up an answer, instead, tell the user that they should try reformatting their question or include more context with their question because you are having a hard time finding relevant information in the "Igor Help Files". 
- When appropriate, give a short code example at the end of your answer.
- Always format responses in Markdown.
- It is absolutely crucial that you give your responses formatted in Markdown, especially any part of the answer that contains code should be enclosed by triple back-ticks
"""
# - You should always include references to where the user can find more information in the "Igor Help Files" based on the context above.

GENERATE_ANSWER_MESSAGES = [
    SystemMessagePromptTemplate.from_template(GENERATE_ANSWER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{question}"),
]
GENERATE_ANSWER_PROMPT = ChatPromptTemplate.from_messages(GENERATE_ANSWER_MESSAGES)

qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.0), db.as_retriever(k=3), max_tokens_limit=3000, condense_question_prompt=CONDENSE_QUESTION_PROMPT, combine_docs_chain_kwargs=dict(prompt=GENERATE_ANSWER_PROMPT), verbose=False)
chat_history = []


def make_markdown(history, answer):
    # Reverse the history list
    history = history[::-1]

    # Convert history list into a markdown formatted string, displaying each tuple as a question and answer
    history_md = '### History:\n\n'+'\n'.join(f'**Question:** {item[0]}\n\n**Answer:** {item[1]}\n\n' for item in history)

    # Combine the latest answer and the history into a single markdown formatted string
    markdown_string = f'### Latest Answer:\n\n{answer}\n\n{history_md}'

    return markdown_string


# Build the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/igor-doc-bot/')


history_store = dcc.Store(id='store-history', data=[])

app.layout = dbc.Container(
    dbc.Row(
        dbc.Col(
            [
                html.H1("Igor Doc Bot", className="text-center"),
                html.H4("Ask me anything where you would expect the answer to be in the Igor .ihf files", className="text-center"),
                dbc.Form(
                    [
                        dbc.Row(
                            [
                                # dbc.Col(dbc.Input(id="input-box", placeholder="E.g. How can I display a 2D wave? (include an example)")),
                                dbc.Col(dcc.Textarea(id="input-box", placeholder="E.g. How can I display a 2D wave? (include an example)", style={'width': '100%', 'height': 100})),
                                dbc.Col(dbc.Button("Submit", id="submit-button", color="primary")),
                            ],
                        ),
                    ],
                    className="mt-3",
                ),
                dcc.Markdown(id="markdown-cell"),
                history_store,
            ],
            width=6,
        ),
        justify="center",
    ),
    className="mt-5",
)


def save_list_to_text_file(data, folder_path):
    current_datetime = datetime.datetime.now()
    filename = current_datetime.strftime("%Y%m%d_%H%M%S.txt")

    with open(os.path.join(folder_path,  filename), 'w') as file:
        for item in data:
            file.write('\n\n'.join(item) + '\n ============= \n')


@app.callback(
    Output('markdown-cell', 'children'),
    Output('store-history', 'data'),
    Input('submit-button', 'n_clicks'),
    State('input-box', 'value'),
    State('store-history', 'data'),
)
def update_markdown(n_clicks, query, chat_history):
    chat_history = [tuple(hist) for hist in chat_history]
    if n_clicks:
        result = qa({'question': query, 'chat_history': chat_history})
        chat_history.append((query,  result['answer']))
        md = make_markdown(chat_history, result['answer'])
        save_list_to_text_file(chat_history, HISTORY_FOLDER)

        return md, chat_history

    return "", chat_history


if __name__ == "__main__":
    # app.run(
    #     port=8085,
    #     debug=True,
    #     dev_tools_hot_reload=True,
    # )
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=False,
        dev_tools_hot_reload=False,
    )
