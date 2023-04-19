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

from transcription import Transcription, DownloadAudio
from summary import Summarize

# whisper
model = whisper.load_model('base')
output = ''
data = []
data_transcription = []
embeddings = []
mp4_video = ''
audio_file = ''

array = []

user_secret = os.getenv("OPENAI_API_KEY")

# Sidebar
with st.sidebar:
    youtube_link = st.text_input(label = ":white[Youtube link]",
                                placeholder = "")
    st.markdown("OR")
    pdf_file = st.file_uploader(label = ":white[PDF file]",
                                type = "pdf")
    if youtube_link:
        youtube_video = YouTube(youtube_link)
        video_id = pytube.extract.video_id(youtube_link)
        streams = youtube_video.streams.filter(only_audio=True)
        stream = streams.first()
        
        if st.button("Start Analysis"):
            te = Transcription(youtube_link)
            
                
            with st.spinner('Running process...'):
                # # Get the video mp4
                # mp4_video = stream.download(filename='youtube_video.mp4')
                # audio_file = open(mp4_video, 'rb')
                # st.write(youtube_video.title)
                # st.video(youtube_link) 

                # # Whisper
                # output = model.transcribe("youtube_video.mp4")
                
                # # Transcription
                # transcription = {
                #     "title": youtube_video.title.strip(),
                #     "transcription": output['text']
                # }
                transcribed_data = te.transcribe()

                data_transcription.append(transcribed_data)
                pd.DataFrame(data_transcription).to_csv('transcription.csv') 
                segments = transcribed_data['segments']
    
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
                pd.DataFrame(data).to_csv('word_embeddings.csv') 
                st.success('Analysis completed')

st.markdown('# Almithal')

DEFAULT_WIDTH = 80

width = 40

width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Introduciton", "Summary", "Transcription", "Mind Map", "Key Questions", "Q&A"])
with tab1:
    st.markdown("# How do I use this?")
with tab2: 
    st.header("Summary:")
with tab3:
    st.header("Transcription")
with tab4:
    st.markdown("## Mind Map")
with tab5:
    st.header("Key Questions:")
    if os.path.exists("word_embeddings.csv"):
        df = pd.read_csv('word_embeddings.csv')
        st.write(df)
with tab6:
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

