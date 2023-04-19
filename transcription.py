# For downloading from youtube and transcribing audio
from pytube import YouTube
from moviepy.editor import * 
from pydub import AudioSegment
from pydub.utils import make_chunks

# For getting text from PDF
import PyPDF2

# For transcription
import openai, whisper

# For other stuff
import os, re
import time, math

# USEFUL CONSTANTS

# Duration is set to 6 minutes = 360 seconds = 360000 milliseconds
DURATION = 360000

# Maximum audio file size is 18MB
MAX_FILE_SIZE_BYTES = 18000000

# The model to use for transcription
WHISPER_MODEL = "tiny"

class DownloadAudio:
    """Downloads the audio from a youtube video and saves it to multiple .wav files in the specified folder"""

    def __init__(self, link) -> None:
        self.link = link
        self.yt = YouTube(self.link)
        self.YOUTUBE_VIDEO_ID = link.split("=")[1]
        self.WAV_FILE_NAME = f"{self.YOUTUBE_VIDEO_ID}.wav"

    def get_yt_title(self) -> str:
        """Returns the title of the youtube video"""
        while True:
            try:
                title = self.yt.title
                return title
            except:
                print("Failed to get name. Retrying...")
                time.sleep(1)
                self.yt = YouTube(self.link)
                continue

    def download(self, pathname:str) -> list[str]:
        """
        Download the audio from the youtube video and saves it to multiple .wav files
        in the specified folder. Returns a list of the paths to the .wav files.
        """

        # Check if the folder for the VIDEO_ID exists
        if not os.path.exists(pathname):
            os.mkdir(pathname)

        # Download the .mp4 file
        audiostream = self.yt.streams.filter(only_audio=True).first()
        outfile_path = audiostream.download(pathname)

        # Convert the .mp4 file to .wav
        FINAL_WAV_PATH = f"{pathname}/{self.WAV_FILE_NAME}"
        wav_file = AudioFileClip(outfile_path)
        wav_file.write_audiofile(FINAL_WAV_PATH, bitrate="16k", fps=16000)

        # Load the input .wav file
        audio = AudioSegment.from_wav(FINAL_WAV_PATH)
    
        # Get the duration of the input file in milliseconds
        total_byte_size = os.path.getsize(FINAL_WAV_PATH)
        
        # If the total duration is less than the duration of each segment,
        # then just return the original file
        if total_byte_size < MAX_FILE_SIZE_BYTES:
            return [FINAL_WAV_PATH]

        # Get the size of the wav file
        channels = audio.channels
        sample_width = audio.sample_width
        duration_in_sec = math.ceil(len(audio) / 1000)
        sample_rate = audio.frame_rate
        bit_rate = sample_width * 8
        wav_file_size = (sample_rate * bit_rate * channels * duration_in_sec) / 8

        # Get the length of each chunk in milliseconds and make the chunks
        chunk_length_in_sec = math.ceil((duration_in_sec * MAX_FILE_SIZE_BYTES ) / wav_file_size)   #in sec
        chunk_length_ms = chunk_length_in_sec * 1000
        chunks = make_chunks(audio, chunk_length_ms)

        # Export all of the individual chunks as wav files
        chunk_names = []
        for i, chunk in enumerate(chunks):
            chunk_name = f"{self.YOUTUBE_VIDEO_ID}_{i}.wav"
            output_chunk_path = f"{pathname}/{chunk_name}"
            chunk_names.append(output_chunk_path)
            chunk.export(f"{output_chunk_path}", format="wav")
        
        return chunk_names


class Transcription:
    """Performs transcription on a PDF or a link to a youtube video"""

    def __init__(self, datalink) -> None:
        self.datalink = datalink
        self.model = whisper.load_model('tiny')
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    def transcribe(self) -> dict:
        """Returns the transcription of the PDF or youtube video as a string"""

        start_time = time.time()
        if self.datalink.startswith("http"):
            transcript = self.get_text_from_link()
        else:
            transcript = self.get_text_from_pdf()
        end_time = time.time()
        print(f"transcription took {end_time - start_time} seconds")
        return transcript
            

    def get_text_from_pdf(self) -> dict:
        # Get the text from the PDF
        with open(self.datalink, 'rb') as f:

            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    
                    # Preprocess the text
                    page_text = re.sub(r'\n', ' ', page_text)  # remove newlines
                    page_text = re.sub(r'\s+', ' ', page_text)  # remove extra spaces
                    
                    # Add to the overall text
                    text += page_text

        # Construct the final output
        output = {
            "name:": self.datalink,
            "transcription": text,
            "segments": None
        }
        return output


    def get_text_from_link(self) -> dict:
        # Get the audio file
        audio_file = DownloadAudio(self.datalink)

        # Get the names of the stored wav files
        YOUTUBE_VIDEO_ID = self.datalink.split("=")[1]
        FOLDER_NAME = f"./tests/{YOUTUBE_VIDEO_ID}"

        # Get the names of the stored wav files
        file_names = audio_file.download(FOLDER_NAME)

        # # Get the transcription of each segment
        text_transcriptions = ""
        segments = []
        for file_name in file_names:
            
            audio = open(file_name, "rb")

            # Get the transcription
            # We are guaranteed that this will be under the max size
            chunk_transcription = self.model.transcribe(file_name)
            text_transcriptions += chunk_transcription["text"]
            segments.append(chunk_transcription["segments"])

        final_transcription = {
            "title:": audio_file.get_yt_title(),
            "transcription": text_transcriptions,
            "segments": segments
        }

        return final_transcription