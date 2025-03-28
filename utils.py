import os
import re
import base64
import cohere
import whisper
import streamlit as st
from dotenv import load_dotenv
import json
from elevenlabs import generate, save, set_api_key

# Load environment variables
load_dotenv()
api_key = os.getenv("cohere_api_key")

# Initialize Cohere client
client = cohere.ClientV2(api_key=api_key)

# Set Eleven Labs API key
set_api_key("")


def get_answer(messages):
    with open('custom_qa.json') as f:
        facts = json.load(f)
    system_message = [{
        "role": "system",
        "content": (
            "You are Shaw Alam, not an AI, a human aged 26 who is calm and detail-oriented. "
            f"You always keep in mind: {facts['known_facts']}"
            "Always stay true to how Shaw Alam would talk, think, and express."
        )
    }]
    trimmed_messages = messages[-3:] if len(messages) > 3 else messages

    messages = system_message + trimmed_messages
    response = client.chat(
        model="command-a-03-2025",
        messages=messages
    )
    return response.message.content[0].text


def speech_to_text(audio_data):
    model = whisper.load_model("base")
    result = model.transcribe(audio_data)
    return result["text"]


def clean_text(text):
    """Remove Markdown formatting like **bold**, *italic*, __underline__, etc."""
    return re.sub(r'[*_`]', '', text)


def text_to_speech(text):
    input_text = clean_text(text)

    # Generate the audio (streamed)
    audio = generate(
        text=input_text,
        voice="Roger",  # Change to any available Eleven Labs voice
        model="eleven_monolingual_v1"
    )

    # Save the audio
    mp3_file_path = "temp_audio_play.mp3"
    save(audio, mp3_file_path)

    return mp3_file_path


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
