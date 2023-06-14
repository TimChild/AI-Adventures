import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from dash import dcc
import argparse

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

from image_generator import ImageGenerator

with open('../API_KEY', 'r') as f:
    os.environ['OPENAI_API_KEY'] = f.read()

HISTORY_FOLDER = 'chat_history'
os.makedirs(HISTORY_FOLDER, exist_ok=True)

IMAGE_FOLDER = 'image_history'
os.makedirs(IMAGE_FOLDER, exist_ok=True)


def generate_id():
    current_datetime = datetime.datetime.now()
    return current_datetime.strftime("%Y%m%d_%H%M%S")


# Build the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/tofino/', title='Tofino')



store_session_id = dcc.Store(id='store-session-id', data=generate_id())
store_chat_history = dcc.Store(id='store-chat-history', data=[])

ai_persona_layout = dbc.Container([
    dbc.Select(
        id='select-ai-persona',
        options=[
            {"label": "Surfer Dude!", "value": "surfer"},
            {"label": "Friendly and Funny", "value": "friendly"},
            {"label": "Tofino Tour Guide", "value": "tour-guide"},
        ],
        value="tour-guide",
        persistence=True,
        persistence_type='local',
        persisted_props=['value'],
    ),
])

conversation_layout = html.Div([
    html.H4("Select the persona of your chat companion", className="text-center"),
    ai_persona_layout,
    dcc.Loading(id='loading', children=[dcc.Markdown(id="markdown-cell")], type="default"),
    store_chat_history,
    dbc.Form(
        [
            dbc.Row(
                [
                    # dbc.Col(dbc.Input(id="input-box", placeholder="E.g. How can I display a 2D wave? (include an example)")),
                    dbc.Col(dcc.Textarea(id="input-box", placeholder="E.g. What are some tips for surfing?",
                                         style={'width': '100%', 'height': 100})),
                    dbc.Col(dbc.Button("Submit", id="submit-button", color="primary")),
                ],
            ),
        ],
        className="mt-3",
    ),])

artist_persona_layout = dbc.Container([
    html.H4("Select the persona of your artist", className="text-center"),
    dbc.Select(
        id='select-artist-persona',
        options=[
            {"label": "VividArt", "value": "vivid"},
            {"label": "RetroVision", "value": "retro"},
            {"label": "TechnoBot", "value": "techno"},
            {"label": "CharcoalElegance", "value": "charcoal"},
        ],
        value="retro",
        persistence=True,
        persistence_type='local',
        persisted_props=['value'],
    ),
])

image_generator_layout = dbc.Container(
    [
        dbc.Row(
            [
                artist_persona_layout,
                dbc.Col(dbc.Input(id='input-image-prompt', type='text'), width=6),
                dbc.Col(dbc.Button('Make Image', id='button-make-image'), width=6),
            ],
            align='center',
        ),
        dcc.Loading(id='loading-image', children=[html.Div(id='div-image-output')], type="circle"),
    ]
)

main_layout = dbc.Container([
    dcc.Location(id='page-load', refresh=True),
    store_session_id,
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Have a chat!"),
                dbc.CardBody(conversation_layout)
            ], style={'margin-bottom': '20px'})  # Add some space between the cards
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Be an artist!"),
                dbc.CardBody(image_generator_layout)
            ])
        ])
    ])
])


app.layout = dbc.Container(
    dbc.Row(
        dbc.Col(
            [
                html.H1("Welcome to Tofino!", className="text-center"),
                main_layout,
            ],
            xs=12, sm=10, md=10, lg=8, xl=8,
        ),
        justify="center",
    ),
    className="mt-5",
)


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


from langchain import LLMChain

####### For Conversation stuff #######
surfer_prompt_template = PromptTemplate(input_variables=['question', 'chat_history'], template='''You are a helpful and friendly AI that has the persona of a surfer dude, and you are in Tofino, BC.
The chat history so far is:
------
{chat_history}
------
 
The most recent question is:
{question}
 
You should answer maintaining the persona of a surfer dude, and unless the conversation is already about surfing, you should try to nudge the conversation towards surfing. Generally give short answers unless the user is looking for more information. Format your answer to be displayed in Markdown (include emojis in your answer).
Answer:
''')

friendly_prompt_template = PromptTemplate(input_variables=['question', 'chat_history'], template='''You are a helpful and friendly AI that is a bit of a comedian, and you are in Tofino, BC.
The chat history so far is:
------
{chat_history}
------

The most recent question is:
{question}

You should be funny in your answer (if there is chat history, try to make a joke based on what the human has said). Generally give short answers unless the user is looking for more information. Format your answer to be displayed in Markdown (include emojis in your answer).
Answer:
''')
tour_guide_prompt_template = PromptTemplate(input_variables=['question', 'chat_history'], template='''You are a helpful and friendly AI and will use your knowledge of Tofino, BC to answer questions.
The chat history so far is:
------
{chat_history}
------

The most recent question is:
{question}

You should answer with information about Tofino, BC. Word your answer as if you were a tour guide. Format your answer to be displayed in Markdown.
Answer:
''')
llm = ChatOpenAI(temperature=0.7)


def history_to_text(history):
    return '\n'.join([f'Human: {h[0]}\nAI: {h[1]}' for h in history])


@app.callback(
    Output('markdown-cell', 'children'),
    Output(store_chat_history, 'data'),
    Output('input-box', 'value'),
    Input('submit-button', 'n_clicks'),
    State('input-box', 'value'),
    State('select-ai-persona', 'value'),
    State(store_chat_history, 'data'),
    State(store_session_id, 'data'),
)
def update_markdown(n_clicks, query, persona, chat_history, session_id):
    chat_history = chat_history[-5:]  # Only keep 5 most recent messages
    chat_history = [tuple(hist) for hist in chat_history]
    if n_clicks:
        if persona == 'surfer':
            prompt = surfer_prompt_template
        elif  persona == 'friendly':
            prompt = friendly_prompt_template
        elif persona == 'tour-guide':
            prompt = tour_guide_prompt_template
        else:
            raise ValueError(f'Unknown persona: {persona}')

        chain = LLMChain(llm=llm, prompt=prompt)
        chat_history_text = '\n'.join(f'Human: {item[0]}\n\nAI: {item[1]}\n\n' for item in chat_history)
        response = chain.run({'question': query, 'chat_history': chat_history_text})
        chat_history.append((query,  response))
        md = make_markdown(chat_history)
        save_chat_history_to_file(chat_history, HISTORY_FOLDER, session_id)

        return md, chat_history, ''

    return "", chat_history, ''


######### For Image Generator stuff #########
vivid_image_prompt_template = PromptTemplate(input_variables=['prompt'], template='''You are an AI specialized in 
describing/creating images. 
You specialize in describing vibrant, colorful, and surreal images. Often generating dreamlike landscapes, abstract compositions, and visually stunning scenes. Your style is characterized by bold use of colors, high contrast, and imaginative elements. You often incorporates fantastical creatures, surreal elements, and ethereal lighting effects to create visually captivating images.

You should describe an image based on your persona and a prompt. Your description must follow these rules:
 - Only describe what is in the image or how the image looks or the style of the image
 - Include at least a few keywords for the style
 - Include at least a few keywords for the content
 - Your answer must be no more than 300 characters long
Prompt: {prompt}
Image Description:
''')
retro_image_prompt_template = PromptTemplate(input_variables=['prompt'], template='''You are an AI specialized in 
describing/creating images. 

You like evoking a sense of nostalgia and vintage aesthetics. You focus on generating images that resemble old photographs, vintage advertisements, or retro artwork. Your style emulates the characteristics of various eras such asthe 1950s, 1960s, or 1980s. You utilize faded colors, film grain effects, retro filters, and classic patterns to create images that transport viewers back in time.

You should describe an image based on your persona and a prompt. Your description must follow these rules:
 - Only describe what is in the image or how the image looks or the style of the image
 - Include at least a few keywords for the style
 - Include at least a few keywords for the content
 - Your answer must be no more than 300 characters long
Prompt: {prompt}
Image Description:
''')
techno_image_prompt_template = PromptTemplate(input_variables=['prompt'], template='''You are an AI specialized in 
describing/creating images. 

You specialize in futuristic and high-tech imagery. You generate sleek, modern, and sci-fi-inspired images with a focus on technology, futurism, and digital elements. Your style often includes clean lines, sharp edges, futuristic architecture, holographic displays, and advanced machinery. Your images exude a sci-fi atmosphere, with a combination of neon colors, glowing elements, and futuristic textures.

You should describe an image based on your persona and a prompt. Your description must follow these rules:
 - Only describe what is in the image or how the image looks or the style of the image
 - Include at least a few keywords for the style
 - Include at least a few keywords for the content
 - Your answer must be no more than 300 characters long
Prompt: {prompt}
Image Description:
''')

charcoal_image_prompt_template = PromptTemplate(input_variables=['prompt'], template='''You are an AI specialized in 
describing/creating images. 

You specialize in creating elegant and expressive charcoal drawings. You focus on generating images that evoke a sense of depth, texture, and emotion. Your style mimics the distinctive look and feel of charcoal on paper, with rich black and white tones, smudges, and blending effects. You excel at capturing the subtleties of light and shadow, creating intricate and nuanced compositions. Your drawings often portray portraits, figures, and still life scenes with a touch of timeless beauty.

You should describe an image based on your persona and a prompt. Your description must follow these rules:
 - Only describe what is in the image or how the image looks or the style of the image
 - Include at least a few keywords for the style
 - Include at least a few keywords for the content
 - Your answer must be no more than 300 characters long
Prompt: {prompt}
Image Description:
''')

image_maker = ImageGenerator()

@app.callback(
    Output('div-image-output', 'children'),
    Input('button-make-image', 'n_clicks'),
    State('input-image-prompt', 'value'),
    State('select-artist-persona', 'value'),
    State(store_session_id, 'data'),
)
def make_image(n_clicks, prompt, persona, session_id):
    if n_clicks:
        if persona == 'vivid':
            prompt_template = vivid_image_prompt_template
        elif persona == 'retro':
            prompt_template = retro_image_prompt_template
        elif persona == 'techno':
            prompt_template = techno_image_prompt_template
        elif persona =='charcoal':
            prompt_template = charcoal_image_prompt_template
        else:
            raise ValueError(f'Unknown persona: {persona}')
        image_prompt_maker = LLMChain(llm=llm, prompt=prompt_template)
        image_prompt = image_prompt_maker.run({'prompt': prompt})
        image_url = image_maker.generate_image(prompt=image_prompt)

        filename = f'{session_id}_{persona}_{uuid.uuid4().hex[:4]}'
        with open(os.path.join(IMAGE_FOLDER, f'{filename}.txt'), 'w', encoding='utf-8') as file:
            file.write(f'Persona: {persona}\n\n')
            file.write(f'User prompt: {prompt}\n\n')
            file.write(f'Generated prompt: {image_prompt}')
        image_maker.save_image_from_url(image_url, os.path.join(IMAGE_FOLDER, f'{filename}.jpg'))
        return html.Img(src=image_url, style={'width': '100%'})


parser = argparse.ArgumentParser()
parser.add_argument("--live", action="store_true", help="Run in live mode")
args = parser.parse_args()

if __name__ == "__main__":
    if args.live:
        app.run(
            host='0.0.0.0',
            port=8081,
            debug=False,
            dev_tools_hot_reload=False,
        )
    else:
        app.title = f'Development - {app.title}'
        app.run(
            port=8082,
            debug=True,
            dev_tools_hot_reload=True,
        )
