import openai

with open(r'D:\GitHub\ai_adventures\API_KEY', "r") as f:
    key = f.read()

openai.api_key = key
