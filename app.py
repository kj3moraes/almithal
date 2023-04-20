import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

import pandas as pd
import numpy as np
import whisper
from streamlit_chat import message
import openai
from openai.embeddings_utils import distances_from_embeddings
import os, json
import math

from transcription import *
from keywords import KeyTakeaways
from summary import TextSummarizer
import models as md

# whisper
model = whisper.load_model('base')
output = ''
data = []
data_transcription = {"title":"", "text":""}
embeddings = []
audio_file = ''
folder_name = "./tests/"
input_accepted = False

is_completed_analysis = False

config = Config(height=500,
                width=700, 
                directed=True, 
                collapsible=True)

nodes = []
edges = []

nodes.append( Node(id="spiderman", 
                   label="Peter Parker", 
                   size=25, 
                   shape="circularImage",
                   image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_spiderman.png") 
            ) # includes **kwargs
nodes.append( Node(id="captain_marvel", 
                   label="Captain Marvel",
                   size=25,
                   shape="circularImage",
                   image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png") 
            )
edges.append( Edge(source="captain_marvel", 
                   label="friend_of", 
                   target="spiderman", 
                   # **kwargs
                   ) 
            )

user_secret = os.getenv("OPENAI_API_KEY")

# Define the purpose of the application
st.header('Almithal')
st.subheader('Almithal is a comprhensive video and PDF study buddy.')
st.write('It provides a summary, transcription, key insights, a mind map and a Q&A feature where you can actually "talk" to the datasource.')

bar = st.progress(0)

# =========== SIDEBAR FOR GENERATION ===========
with st.sidebar:
    youtube_link = st.text_input(label = ":white[Youtube link]",
                                placeholder = "")
    # st.markdown("OR")
    pdf_file = None
    
    gen_keywords = st.radio(
        "Generate keywords from text?",
        ('Yes', 'No')
    )
    
    gen_transcript = st.radio(
        "Generate transcript ?",
        ('Yes', 'No')
    )

    gen_summary = st.radio(
        "Generate summary from text? (recommended for label matching below, but will take longer)",
        ('Yes', 'No')
    )
    
    if youtube_link:
        input_accepted = True
        vte = VideoTranscription(youtube_link)
        YOUTUBE_VIDEO_ID = youtube_link.split("=")[1]
        folder_name = f"./tests/{YOUTUBE_VIDEO_ID}"
        
        if st.button("Start Analysis"):
            with st.spinner('Running process...'):
                
                if not os.path.exists(f'{folder_name}/data_transcription.json'):
                    # Get the video wav
                    data_transcription = vte.transcribe()

                    with open(f"{folder_name}/data_transcription.json", "w") as f:
                        json.dump(data_transcription, f, indent=4)
                else:
                    with open(f"{folder_name}/data_transcription.json", "r") as f:
                        data_transcription = json.load(f)
                    
                segments = data_transcription['segments']
                
                # Generate embeddings
                if not os.path.exists(f"{folder_name}/word_embeddings.csv"):
                    for i, segment in enumerate(segments):
                        bar.progress(max(math.ceil((i/len(segments) * 100)), 1))
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
                    
                    pd.DataFrame(data).to_csv(f'{folder_name}/word_embeddings.csv') 
                else:   
                    data = pd.read_csv(f'{folder_name}/word_embeddings.csv')
                    embeddings = data["embedding"]
                bar.progress(100)
                st.success('Analysis completed')  
                
    # PDF Transcription 
    elif pdf_file is not None:
        pte = PDFTranscription(pdf_file.name)
        folder_name = f"./tests/{pdf_file.name}".replace(' ', '')
        
        if st.button("Start Analysis"):
            with st.spinner('Running process...'):
                
                if not os.path.exists(f'{folder_name}/data_transcription.json'):
                    # Get the video wav
                    data_transcription = pte.transcribe(pdf_file)

                    with open(f"{folder_name}/data_transcription.json", "w") as f:
                        json.dump(data_transcription, f, indent=4)
                else:
                    with open(f"{folder_name}/data_transcription.json", "r") as f:
                        data_transcription = json.load(f)
                    
                segments = data_transcription['segments']
                
                # Generate embeddings
                if not os.path.exists(f"{folder_name}/word_embeddings.csv"):
                    for i, segment in enumerate(segments):
                        bar.progress(max(math.ceil((i/len(segments) * 100)), 1))
                        openai.api_key = user_secret
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
                    embeddings = data["embedding"]
                bar.progress(100)
                st.success('Analysis completed')  
    text_df = pd.DataFrame.from_dict({"title": [data_transcription["title"]], "text":[data_transcription["text"]]})
            

with st.spinner('Breaking up the text and doing analysis...'):
    # For each body of text, create text chunks of a certain token size required for the transformer
    text_chunks_lib = dict()
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
    key_engine = KeyTakeaways(title_entry)
    keywords = key_engine.get_keywords(text_chunks_lib)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Introduction", "Summary", "Transcription", "Mind Map", "Keywords", "Q&A"])

# =========== INTRODUCTION ===========
with tab1:
    st.markdown("## How do I use this?")
    st.markdown("Do one of the following")
    st.markdown('* Type in your youtube URL that you want worked on')
    st.markdown('* Place the PDF file that you want worked on')
    st.markdown("**Once the file / url has finished saving, a 'Start Analysis' button will appear. Click on this button to begin the note generation**")
    st.warning("NOTE: This is just a demo product in alpha testing. Any and all bugs will soon be fixed")

# =========== SUMMARIZATION ===========
with tab2: 
    st.header("Summary")
    if input_accepted:
        if gen_summary == 'Yes':
            with st.spinner("Generating summary ...."):
                # text_transcription = data_transcription['text']
                # print("Working on summarization")
                # se = TextSummarizer()
                # summary = se.summarize(text_chunks_lib)
                st.write("smmary daone")
        else:
            st.warning("Summary was not selected")

# =========== TRANSCRIPTION ===========
with tab3:
    if input_accepted:
        st.header("Transcription")
        if gen_transcript == 'Yes':
            with st.spinner("Generating transcript ..."):
                for text in text_chunks_lib[title_entry]:
                    st.write(text)
        else:
            st.warning("Transcription was not selected")
    else:
        st.error("You need to give a data source")

# =========== MIND MAP ===========
with tab4:
    st.header("Mind Map")
    
    return_value = agraph(nodes=nodes, 
                        edges=edges, 
                        config=config)

# =========== KEY TAKEAWAYS ===========
with tab5:
    st.header("Keywords:")
    for i, keyword in enumerate(keywords):
        st.markdown(f"{i+1}. {keyword}")
    
# =========== QUERY BOT ===========
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

    def get_embedding_text(prompt):
        openai.api_key = user_secret
        response = openai.Embedding.create(
            input= prompt.strip(),
            model="text-embedding-ada-002"
        )
        q_embedding = response['data'][0]['embedding']
        df=pd.read_csv(f'{folder_name}/word_embeddings.csv', index_col=0)
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)

        df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
        returns = []
        
        # Sort by distance with 2 hints
        for i, row in df.sort_values('distances', ascending=True).head(4).iterrows():
            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        return "\n\n###\n\n".join(returns)

    def generate_response(prompt):
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
            temperature=0.5,
        )
        message = completions.choices[0].text
        return message

    if user_input:
        text_embedding = get_embedding_text(user_input)
        with open(f'{folder_name}/data_transcription.json', "r") as f:
            title = json.load(f)['title']
        string_title = "\n\n###\n\n".join(title)
        user_input_embedding = 'Using this context: "'+string_title+'. '+text_embedding+'", answer the following question. \n'+user_input
        # st.write(user_input_embedding)
        output = generate_response(user_input_embedding)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            

if is_completed_analysis:
    st.header("Key Takeaways:")