import os
import openai

import graphviz
import streamlit as st

class MindMap:
    
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
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

        st.graphviz_chart(graph)