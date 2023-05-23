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


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container(
    dbc.Row(
        dbc.Col(
            [
                html.H1("Igor Doc Bot", className="text-center"),
                dbc.Form(
                    [
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="input-box", placeholder="Enter text...")),
                                dbc.Col(dbc.Button("Submit", id="submit-button", color="primary")),
                            ],
                        ),
                    ],
                    className="mt-3",
                ),
                dcc.Markdown(id="markdown-cell"),
            ],
            width=6,
        ),
        justify="center",
    ),
    className="mt-5",
)


embedding = OpenAIEmbeddings()
db = FAISS.load_local("faiss_dbs", embeddings=embedding, index_name="igor-test")

# TODO: The chat history should be stored in a ServerSide store using the dash-extensions package
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.0), db.as_retriever())
chat_history = []


def make_markdown(history, answer):
    # Reverse the history list
    history = history[::-1]

    # Convert history list into a markdown formatted string, displaying each tuple as a question and answer
    history_md = '### History:\n\n'+'\n'.join(f'**Question:** {item[0]}\n\n**Answer:** {item[1]}\n\n' for item in history)

    # Combine the latest answer and the history into a single markdown formatted string
    markdown_string = f'### Latest Answer:\n\n{answer}\n\n{history_md}'

    return markdown_string


@app.callback(
    Output('markdown-cell', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-box', 'value'),
)
def update_markdown(n_clicks, query):
    if n_clicks:
        result = qa({'question': query, 'chat_history': chat_history})
        chat_history.append((query,  result['answer']))
        md = make_markdown(chat_history, result['answer'])
        return md

    return ""






if __name__ == "__main__":
    app.run(
        debug=True,
        dev_tools_hot_reload=True,
    )
