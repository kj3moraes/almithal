import os
import openai

import graphviz
import streamlit as st

class MindMap:
    
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def get_connections(self, text_chunks_libs:dict) -> list:
        
        state_prompt = open("./prompts/mindmap.prompt")
        PROMPT = state_prompt.read()
        state_prompt.close()
        
        
        for key in text_chunks_libs:
            for text_chunk in text_chunks_libs[key]:
                PROMPT = PROMPT.replace("$prompt", text_chunk)
                
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt = PROMPT,
                    temperature=0.5,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                
                print("response is ", response.choices[0].text)
                
                
        return []
            
        
    def generate_graph(self, text_chunks_libs:dict):
        graph = graphviz.Digraph()
        graph.edge('run', 'intr')
        graph.edge('intr', 'runbl')
        graph.edge('runbl', 'run')
        graph.edge('run', 'kernel')
        graph.edge('kernel', 'zombie')
        graph.edge('kernel', 'sleep')
        graph.edge('kernel', 'runmem')
        graph.edge('sleep', 'swap')
        graph.edge('swap', 'runswap')
        graph.edge('runswap', 'new')
        graph.edge('runswap', 'runmem')
        graph.edge('new', 'runmem')
        graph.edge('sleep', 'runmem')
        self.get_connections(text_chunks_libs)
        st.graphviz_chart(graph)