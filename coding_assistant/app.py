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
import uuid
import datetime


with open('../API_KEY', 'r') as f:
    os.environ['OPENAI_API_KEY'] = f.read()

HISTORY_FOLDER = 'chat_history'
os.makedirs(HISTORY_FOLDER, exist_ok=True)


def generate_id():
    current_datetime = datetime.datetime.now()
    return current_datetime.strftime("%Y%m%d_%H%M%S")


# Build the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/pythia/', title='Pythia')


store_session_id = dcc.Store(id='store-session-id', data=generate_id())
store_chat_history = dcc.Store(id='store-chat-history', data=[])

session_tracker_layout = html.Div([
    dcc.Location(id='page-load', refresh=True),
    store_session_id,
    ])

# Define the text inputs
text_prompt_input = html.Div(
    [
        dbc.Label("Enter General Text Prompt", html_for='text-prompt-input'),
        dcc.Textarea(id='text-prompt-input', style={'width': '100%', 'height': 100}),
    ],
    className="mb-3",
)

existing_code_input = html.Div(
    [
        dbc.Label("Copy in Existing Code", html_for='existing-code-input'),
        dcc.Textarea(id='existing-code-input', style={'width': '100%', 'height': 100}),
    ],
    className="mb-3",
)

error_messages_input = html.Div(
    [
        dbc.Label("Copy in Error Messages", html_for='error-messages-input'),
        dcc.Textarea(id='error-messages-input', style={'width': '100%', 'height': 100}),
    ],
    className="mb-3",
)

inputs_layout = dbc.Form([text_prompt_input, existing_code_input, error_messages_input])

# Define the options
options_button = dbc.Button("Options/Preferences", id="open", className="mr-1")

options_modal = dbc.Modal(
    [
        dbc.ModalHeader("Options/Preferences"),
        dbc.ModalBody(dcc.Input(id="input")),
        dbc.ModalFooter(
            dbc.Button("Close", id="close", className="ml-auto")
        ),
    ],
    id="modal",
)

options_layout = html.Div([options_button, options_modal])

inputs_card = dbc.Card(
    [
        dbc.CardHeader("Inputs"),
        dbc.CardBody([inputs_layout, options_layout]),
    ],
    className="mb-3",
)


# Define the display outputs
generated_code_display = dbc.Card(
    [
        dbc.CardHeader("Generated Code"),
        dbc.CardBody(dcc.Markdown(id='generated-code-display')),
    ],
    className="mb-3",
)

summaries_display = dbc.Card(
    [
        dbc.CardHeader("Summaries"),
        dbc.CardBody(dcc.Markdown(id='summaries-display')),
    ],
    className="mb-3",
)

thought_process_display = dbc.Card(
    [
        dbc.CardHeader("Thought Processes"),
        dbc.CardBody(dcc.Markdown(id='thought-process-display')),
    ],
    className="mb-3",
)

outputs_layout = html.Div([generated_code_display, summaries_display, thought_process_display])


# Combine everything into the overall layout
app.layout = html.Div(
    [
        session_tracker_layout,  # Existing variable that should be at the top of the layout
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            inputs_card,
                            xl=6,  # half the width on extra large screens
                        ),
                        dbc.Col(
                            outputs_layout,
                            xl=6,  # half the width on extra large screens
                        ),
                    ]
                ),
            ],
            className="mt-4",
        )
    ]
)


# Define callback
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [dash.dependencies.State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open





@app.callback(
    Output('store-session-id', 'data'),
    Input('page-load', 'pathname'),
)
def update_session_id(pathname):
    return generate_id()


def make_markdown(history):
    md_string = ''
    if history:
        # Convert history list into a markdown formatted string, displaying each tuple as a question and answer
        md_string = '\n'.join(f'**Question:**\n{item[0]}\n\n**Response:**\n{item[1]}\n\n' for item in history)
        md_string += '\n\n---'
    return md_string


def save_chat_history_to_file(chat_history, folder_path, session_id):
    filename = f'{session_id}.txt'

    with open(os.path.join(folder_path,  filename), 'w', encoding='utf-8') as file:
        for item in chat_history:
            file.write('\n\n'.join(item) + '\n ============= \n')



def history_to_text(history):
    return '\n'.join([f'Human: {h[0]}\nAI: {h[1]}' for h in history])



if __name__ == "__main__":
    app.run(
        port=8083,
        debug=True,
        dev_tools_hot_reload=True,
    )
    # app.run(
    #     host='0.0.0.0',
    #     port=8081,
    #     debug=False,
    #     dev_tools_hot_reload=False,
    # )
