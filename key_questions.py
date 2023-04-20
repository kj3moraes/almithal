import openai
import os
class KeyQuestions:
    
    def __init__(self, keywords:list[str]):
        self.keywords = keywords
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def generate_key_questions(self, text_chunks_lib:dict) -> list:
        PROMPT = """
        asdsad
        """