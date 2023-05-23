import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from dash import dcc

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

import os

os.environ['OPENAI_API_KEY'] = "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"

embedding = OpenAIEmbeddings()
db = FAISS.load_local("faiss_dbs", embeddings=embedding, index_name="igor-test")

# TODO: The chat history should be stored in a ServerSide store using the dash-extensions package
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.0), db.as_retriever(k=3))
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
