# Streamlit classes
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_chat import message

# Data manipulation and embeddings
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings
import whisper

# Exec tasks
import os, json
import math
import re
from threading import Thread

# Custom classes 
from transcription import *
from keywords import Keywords
from summary import TextSummarizer
from takeaways import KeyTakeaways
from mindmap import MindMap
import models as md

def get_initial_message():
    messages=[
            {"role": "system", "content": "You are a helpful AI Tutor. Who anwers brief questions about AI."},
            {"role": "user", "content": "I want to learn AI"},
            {"role": "assistant", "content": "Thats awesome, what do you want to know aboout AI"}
        ]
    return messages

REGEXP_YOUTUBE_URL = "^(https?\:\/\/)?((www\.)?youtube\.com|youtu\.be)\/.+$"

model = whisper.load_model('base')

output = ''
data = []
data_transcription = {"title":"", "text":""}
embeddings = []
text_chunks_lib = dict()
user_input = None
title_entry = None

tldr = ""
summary = ""
takeaways = []
keywords = []

folder_name = "./tests"
input_accepted = False
is_completed_analysis = False
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

user_secret = os.getenv("OPENAI_API_KEY")

# Define the purpose of the application
st.header('Almithal')
st.subheader('Almithal is a comprehensive video and PDF study buddy.')
st.write('It provides a summary, transcription, key insights, a mind map and a Q&A feature where you can actually "talk" to the datasource.')

bar = st.progress(0)

def generate_word_embeddings():
    if not os.path.exists(f"{folder_name}/word_embeddings.csv"):
        for i, segment in enumerate(segments):
            bar.progress(max(math.ceil((i/len(segments) * 50)), 1))
            response = openai.Embedding.create(
                input= segment["text"].strip(),
                model="text-embedding-ada-002"
            )
            embeddings = response['data'][0]['embedding']
            meta = {
                "text": segment["text"].strip(),
                "embedding": embeddings
            }
            data.append(meta)
        
        pd.DataFrame(data).to_csv(f'{folder_name}/word_embeddings.csv') 
    else:   
        data = pd.read_csv(f'{folder_name}/word_embeddings.csv')

  
def generate_text_chunks_lib():
    text_df = pd.DataFrame.from_dict({"title": [data_transcription["title"]], "text":[data_transcription["text"]]})
    input_accepted = True
    
    # For each body of text, create text chunks of a certain token size required for the transformer
    title_entry = text_df['title'][0]
    print(title_entry)
    for i in range(0, len(text_df)):
        nested_sentences = md.create_nest_sentences(document=text_df['text'][i], token_max_length=1024)
        # For each chunk of sentences (within the token max)
        text_chunks = []
        for n in range(0, len(nested_sentences)):
            tc = " ".join(map(str, nested_sentences[n]))
            text_chunks.append(tc)
        
        text_chunks_lib[title_entry] = text_chunks    
    
    # Generate key takeaways 
    key_engine = Keywords(title_entry)
    keywords = key_engine.get_keywords(text_chunks_lib)

# =========== SIDEBAR FOR GENERATION ===========
with st.sidebar:
    youtube_link = st.text_input(label = "Type in your Youtube link", placeholder = "", key="url")
    st.markdown("OR")
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")
    st.markdown("OR")
    audio_file = st.file_uploader("Upload your MP3 audio file", type=["wav", "mp3"])
    
    gen_keywords = st.radio(
        "Generate keywords from text?",
        ('Yes', 'No')
    )

    gen_summary = st.radio(
        "Generate summary from text? (recommended for label matching below, but will take longer)",
        ('Yes', 'No')
    )
    
    if st.button("Start Analysis"):
        
        # Youtube Transcription
        if re.search(REGEXP_YOUTUBE_URL, youtube_link):
            vte = VideoTranscription(youtube_link)
            YOUTUBE_VIDEO_ID = youtube_link.split("=")[1]
            folder_name = f"./tests/{YOUTUBE_VIDEO_ID}"
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            
            with st.spinner('Running transcription...'):
                data_transcription = vte.transcribe()                    
                segments = data_transcription['segments']
                             
        # PDF Transcription 
        elif pdf_file is not None:
            pte = PDFTranscription(pdf_file)
            folder_name = pte.get_redacted_name()
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            
            with st.spinner('Running transcription...'):
                data_transcription = pte.transcribe()
                segments = data_transcription['segments']
        
        # Audio transcription
        elif audio_file is not None:
            ate = AudioTranscription(audio_file)
            folder_name = ate.get_redacted_name()
            if not os.path.exists(f""):
                os.mkdir(folder_name)
            
            with st.spinner('Running transcription...'):
                data_transcription = ate.transcribe()
                segments = data_transcription['segments']
            
            with open(f"{folder_name}/data.json", "w") as f:
                json.dump(data_transcription, f, indent=4)
                
        else:
            st.error("Please type in your youtube link or upload the PDF")  
            st.experimental_rerun()
        
        
        # Generate embeddings
        thread1 = Thread(target=generate_word_embeddings)
        thread1.start()
        # Generate text chunks 
        thread2 = Thread(target=generate_text_chunks_lib)
        thread2.start()
        
        # Wait for them to complete 
        thread1.join()
        thread2.join()
        
        # Generate the summary
        if gen_summary == 'Yes':
            se = TextSummarizer(title_entry)
            text_transcription = data_transcription['text']
            with st.spinner("Generating summary and TLDR..."):
                summary = se.generate_full_summary(text_chunks_lib)
                summary_list = summary.split("\n\n")
                tldr = se.generate_short_summary(summary_list)
        
        # Generate key takeaways
        kt = KeyTakeaways()
        with st.spinner("Generating key takeaways ... "):
            takeaways = kt.generate_key_takeaways(text_chunks_lib)
                
        is_completed_analysis = True
        bar.progress(100)

if is_completed_analysis:
    st.header("Key Takeaways")
    st.write("Here are some of the key takeaways from the data:")
    for takeaway in takeaways:
        st.markdown(f"- {takeaway}")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Introduction", "Summary", "Transcription", "Mind Map", "Keywords", "Q&A"])

# =========== INTRODUCTION ===========
with tab1:
    st.markdown("## How do I use this?")
    st.markdown("Do one of the following")
    st.markdown('* Type in your youtube URL that you want worked on')
    st.markdown('* Place the PDF file that you want worked on')
    st.markdown('* Place the audio file that you want worked on')
    st.markdown("**Once the file / url has finished saving, a 'Start Analysis' button will appear. Click on this button to begin the note generation**")
    st.warning("NOTE: This is just a demo product in alpha testing. Any and all bugs will soon be fixed")
    st.warning("After the note taking is done, you will see multiple tabs for more information")

# =========== SUMMARIZATION ===========
with tab2: 
    if is_completed_analysis:
        st.header("TL;DR")
        for point in tldr:
            st.markdown(f"- {point}")
        st.header("Summary")
        st.write(summary)
    else:
        st.warning("Please wait for the analysis to finish")

# =========== TRANSCRIPTION ===========
with tab3:
    st.header("Transcription")
    if is_completed_analysis:
        with st.spinner("Generating transcript ..."):
            st.write("")
            for text in text_chunks_lib[title_entry]:
                st.write(text)
    else:
        st.warning("Please wait for the analysis to finish")

# =========== MIND MAP ===========
with tab4:
    st.header("Mind Map")
    if is_completed_analysis:
        mindmap = MindMap()
        with st.spinner("Generating mind map..."):
            mindmap.generate_graph(text_chunks_lib)
    else:
        st.warning("Please wait for the analysis to finish")

# =========== KEYWORDS ===========
with tab5:
    st.header("Keywords:")
    if is_completed_analysis and gen_keywords:
        for i, keyword in enumerate(keywords):
            st.markdown(f"{i+1}. {keyword}")
    else:
        st.warning("Please wait for the analysis to finish")

# =========== QUERY BOT ===========
with tab6:
    if is_completed_analysis:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []

        def get_text():
            st.header("Ask me something about the video:")
            input_text = st.text_input("You: ", key="prompt")
            return input_text


        def get_embedding_text(prompt):
            response = openai.Embedding.create(
                input= prompt.strip(),
                model="text-embedding-ada-002"
            )
            q_embedding = response['data'][0]['embedding']
            print("the folder name at got here 1.5 is ", folder_name)
            # df = pd.read_csv(f'{folder_name}/word_embeddings.csv', index_col=0)
            data['embedding'] = data['embedding'].apply(eval).apply(np.array)

            data['distances'] = distances_from_embeddings(q_embedding, data['embedding'].values, distance_metric='cosine')
            returns = []
            
            # Sort by distance with 2 hints
            for i, row in data.sort_values('distances', ascending=True).head(4).iterrows():
                # Else add it to the text that is being returned
                returns.append(row["text"])

            # Return the context
            return "\n\n###\n\n".join(returns)

        def generate_response(prompt):
            one_shot_prompt = '''
                I am YoutubeGPT, a highly intelligent question answering bot.
                If you ask me a question that is rooted in truth, I will give you the answer.
                Q: What is human life expectancy in the United States?
                A: Human life expectancy in the United States is 78 years.
                Q: '''+prompt+'''
                A: 
            '''
            completions = openai.Completion.create(
                engine = "text-davinci-003",
                prompt = one_shot_prompt,
                max_tokens = 1024,
                n = 1,
                stop=["Q:"],
                temperature=0.5,
            )
            message = completions.choices[0].text
            return message
        
        if is_completed_analysis:
            user_input = get_text()
            print("user input is ", user_input)
            print("the folder name at got here 0.5 is ", folder_name)
        else:
            user_input = None
        
        if 'messages' not in st.session_state:
            st.session_state['messages'] = get_initial_message()
        
        if user_input:
            print("got here 1")
            print("the folder name at got here 1.5 is ", folder_name)
            text_embedding = get_embedding_text(user_input)
            print("the folder name at got here 1.5 is ", folder_name)
            print("got here 2")
            title = data_transcription['title']
            string_title = "\n\n###\n\n".join(title)
            user_input_embedding = 'Using this context: "'+string_title+'. '+text_embedding+'", answer the following question. \n'+user_input
            print("got here 3")
            output = generate_response(user_input_embedding)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
            
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


# st.header("What else")