import models as md
import nltk

import openai
import os 

nltk.download("punkt")

class TextSummarizer:
    
    def __init__(self, title):
        self.title = title
        self.model = "gpt-3.5-turbo"
        self.summarizer = md.load_summary_model()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_short_summary(self, summary_chunks:dict) -> list:
        PROMPT =  """
            You are a helpful assistant that summarizes youtube videos.
            Someone has already summarized the video to key points.
            Summarize the key points in at most two sentences that capture the essence of the passage.
        """
        
        final_summary = []
        for summary_chunk in summary_chunks:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": summary_chunk},
                    ],
            )
            summary = response["choices"][0]["message"]["content"]
            final_summary.append(summary)
        
        return final_summary

        
        
    def generate_full_summary(self, text_chunks_lib:dict) -> str:
        sum_dict = dict()
        for _, key in enumerate(text_chunks_lib):
            
            # for key in text_chunks_lib:
            summary = []
            for _, text_chunk in enumerate(text_chunks_lib[key]):
                chunk_summary = md.summarizer_gen(self.summarizer, sequence=text_chunk, maximum_tokens=500, minimum_tokens=100)
                summary.append(chunk_summary)

                # Combine all the summaries into a list and compress into one document, again
                final_summary = "\n\n".join(list(summary))
                sum_dict[key] = [final_summary]

        return sum_dict[self.title][0]
        