import bentoml
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

import models as md
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


class TextSummarizer(bentoml.Runnable):
    
    def __init__(self, title):
        self.title = title
        self.summarizer = md.load_summary_model()
        
    def generate_short_summary(self, text_chunks_libs:pd.DataFrame) -> str:
        PROMPT = """
        You are a as ad .
        """
        
    def generate_full_summary(self, text_chunks_lib:dict) -> str:
        sum_dict = dict()
        for _, key in enumerate(text_chunks_lib):
            
            # for key in text_chunks_lib:
            summary = []
            for num_chunk, text_chunk in enumerate(text_chunks_lib[key]):
                chunk_summary = md.summarizer_gen(self.summarizer, sequence=text_chunk, maximum_tokens=400, minimum_tokens=100)
                summary.append(chunk_summary)

                # Combine all the summaries into a list and compress into one document, again
                final_summary = "\n\n".join(list(summary))
                sum_dict[key] = [final_summary]

        return sum_dict[self.title][0]
        