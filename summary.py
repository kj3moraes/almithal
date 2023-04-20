import bentoml
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

import models as md
import nltk

import openai
import os 

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


class TextSummarizer(bentoml.Runnable):
    
    def __init__(self, title):
        self.title = title
        self.model = "gpt-3.5-turbo"
        self.summarizer = md.load_summary_model()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_short_summary(self, text_chunks_lib:dict) -> str:
        PROMPT = """
        You are a helpful assistant that summarizes youtube videos.
        You are provided chunks of raw audio that were transcribed from the video's audio.
        Summarize the current chunk to succint and clear bullet points of its contents.
        """
        
    def generate_full_summary(self, text_chunks_lib:dict) -> str:
        sum_dict = dict()
        for _, key in enumerate(text_chunks_lib):
            
            # for key in text_chunks_lib:
            summary = []
            for _, text_chunk in enumerate(text_chunks_lib[key]):
                chunk_summary = md.summarizer_gen(self.summarizer, sequence=text_chunk, maximum_tokens=400, minimum_tokens=100)
                summary.append(chunk_summary)

                # Combine all the summaries into a list and compress into one document, again
                final_summary = "\n\n".join(list(summary))
                sum_dict[key] = [final_summary]

        return sum_dict[self.title][0]
        