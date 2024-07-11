from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import librosa
import torch
from transformers import Wave2Vec2ForCTC, Wav2Vec2Tokenizer
import IPython.display as display
from langchain import PromptTemplate, LLMChain, Gemini
import requests
from IPython.display import Audio
import os
import streamlit as st

# load_dotenv(find_dotenv())
# HUGGINGFACEHUB_API_TOKEN = "os.getenv("HUGGINGFACEHUB_API_TOKEN")"
HUGGINGFACEHUB_API_TOKEN = "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"

GOOGLE_API_KEY = "AIzaSyCYH4NNECwXFynFy9KhGs7eBLZ0oQKML4Q"


# speech to text
def speech2text(audio):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wave2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    speech, rate = librosa.load("audio", sr = 16000)

    display.Audio("audio", autoplay = True)
    input_values = tokenizer(speech, return_tensors = 'pt').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim =-1)
    transcriptions = tokenizer.decode(predicted_ids[0])
    print(transcriptions)


# llm

def generate_story(scenario):
    template = """
    You are a chat model;
    You can give answer to the questions asked but the answer should be not more than 250 words;

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm = Gemini(
        model = "gemini-pro", temperature = 1), prompt = prompt, verbose =True)

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story


# text to speech

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/myshell-ai/MeloTTS-English"
    headers = {"Authorization": "Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

scenario = text2speech(audio)
story = generate_story(scenario)
text2speech(story)


if __name__ == '__main__':
    main()