import openai
import os

class Summarize:
    def __init__(self) -> None:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_summary(self, transcript_list: list[str]) -> list[str]:
        PREFIX = "Summarize the following passage:"
        
        summaries = []
        for text in transcript_list:
            response = openai.Completion.create(
                engine="text-ada-001",
                prompt=PREFIX + text,
                temperature=0.4,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.6,
            )

            summaries.append(response.choices[0].text.strip())

        final_summary = "".join(summaries).strip().replace("  ", " ").replace("\n", " ")

        return final_summary