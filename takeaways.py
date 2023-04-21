import openai
import os

class KeyTakeaways:
    
    def __init__(self, keywords:list):
        self.keywords = keywords
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_key_questions(self, text_chunks_lib:dict) -> list:
        PROMPT = """
            You are a super intelligent human and helpful assistant. 
            I am giving you parts of a video transcription that I want to learn from.
            In bullet points, give me at most 4 key takeaways from this text.
        """
        
        final_summary = ""
        
        return "asd"