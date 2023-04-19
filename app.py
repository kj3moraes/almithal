import streamlit as st

import pandas as pd
import numpy as np
import whisper
import pytube
from pytube import YouTube
from streamlit_chat import message
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import os
from dotenv import load_dotenv

# whisper
model = whisper.load_model('base')
output = ''
data = []
data_transcription = []
embeddings = []
mp4_video = ''
audio_file = ''

array = []

# Sidebar
with st.sidebar:
    user_secret = st.text_input(label = ":blue[OpenAI API key]",
                                value="",
                                placeholder = "Paste your openAI API key, sk-",
                                type = "password")
    youtube_link = st.text_input(label = ":red[Youtube link]",
                                value="https://youtu.be/rQeXGvFAJDQ",
                                placeholder = "")
    if youtube_link and user_secret:
        youtube_video = YouTube(youtube_link)
        video_id = pytube.extract.video_id(youtube_link)
        streams = youtube_video.streams.filter(only_audio=True)
        stream = streams.first()
        if st.button("Start Analysis"):
            if os.path.exists("word_embeddings.csv"):
                os.remove("word_embeddings.csv")
                
            with st.spinner('Running process...'):
                # Get the video mp4
                mp4_video = stream.download(filename='youtube_video.mp4')
                audio_file = open(mp4_video, 'rb')
                st.write(youtube_video.title)
                st.video(youtube_link) 

                # Whisper
                output = model.transcribe("youtube_video.mp4")
                
                # Transcription
                transcription = {
                    "title": youtube_video.title.strip(),
                    "transcription": output['text']
                }
                data_transcription.append(transcription)
                pd.DataFrame(data_transcription).to_csv('transcription.csv') 
                segments = output['segments']
    
                #Embeddings
                for segment in segments:
                    openai.api_key = user_secret
                    response = openai.Embedding.create(
                        input= segment["text"].strip(),
                        model="text-embedding-ada-002"
                    )
                    embeddings = response['data'][0]['embedding']
                    meta = {
                        "text": segment["text"].strip(),
                        "start": segment['start'],
                        "end": segment['end'],
                        "embedding": embeddings
                    }
                    data.append(meta)
                # upsert_response = index.upsert(
                #         vectors=data,
                #         namespace=video_id
                #     )
                pd.DataFrame(data).to_csv('word_embeddings.csv') 
                os.remove("youtube_video.mp4")
                st.success('Analysis completed')

st.markdown('# Almithal')

DEFAULT_WIDTH = 80
VIDEO_DATA = "https://youtu.be/bsFXgfbj8Bc"

width = 40

width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

_, container, _ = st.columns([side, 47, side])
container.video(data=VIDEO_DATA)
tab1, tab2, tab3, tab4 = st.tabs(["Introduciton", "Summary", "Transcription", "Q&A"])
with tab1:
    st.markdown("# How do I use this?")
with tab2: 
    st.header("Transcription:")
    if(os.path.exists("youtube_video.mp4")):
        audio_file = open('youtube_video.mp4', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
    if os.path.exists("transcription.csv"):
        df = pd.read_csv('transcription.csv')
        st.write(df)
with tab3:
    st.header("Embedding:")
    if os.path.exists("word_embeddings.csv"):
        df = pd.read_csv('word_embeddings.csv')
        st.write(df)
with tab4:
    user_secret = st.text_input(label = ":blue[OpenAI API key]",
                                placeholder = "Paste your openAI API key, sk-",
                                type = "password")
    st.write('To obtain an API Key you must create an OpenAI account at the following link: https://openai.com/api/')
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text():
        if user_secret:
            st.header("Ask me something about the video:")
            input_text = st.text_input("You: ","", key="input")
            return input_text
    user_input = get_text()

    def get_embedding_text(api_key, prompt):
        openai.api_key = user_secret
        response = openai.Embedding.create(
            input= prompt.strip(),
            model="text-embedding-ada-002"
        )
        q_embedding = response['data'][0]['embedding']
        df=pd.read_csv('word_embeddings.csv', index_col=0)
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)

        df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
        returns = []
        
        # Sort by distance with 2 hints
        for i, row in df.sort_values('distances', ascending=True).head(4).iterrows():
            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        return "\n\n###\n\n".join(returns)

    def generate_response(api_key, prompt):
        one_shot_prompt = '''I am YoutubeGPT, a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.
        Q: What is human life expectancy in the United States?
        A: Human life expectancy in the United States is 78 years.
        Q: '''+prompt+'''
        A: '''
        completions = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = one_shot_prompt,
            max_tokens = 1024,
            n = 1,
            stop=["Q:"],
            temperature=0.2,
        )
        message = completions.choices[0].text
        return message

    if user_input:
        text_embedding = get_embedding_text(user_secret, user_input)
        title = pd.read_csv('transcription.csv')['title']
        string_title = "\n\n###\n\n".join(title)
        user_input_embedding = 'Using this context: "'+string_title+'. '+text_embedding+'", answer the following question. \n'+user_input
        # st.write(user_input_embedding)
        output = generate_response(user_secret, user_input_embedding)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

