# For downloading from youtube and transcribing audio
from pytube import YouTube
from moviepy.editor import * 
from pydub import AudioSegment
from pydub.utils import make_chunks

# For getting text from PDF
from io import StringIO 
from zipfile import ZipFile
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import base64
import streamlit as st
# For transcription
import openai, whisper, torch
import tiktoken

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
        FINAL_WAV_PATH = f"{pathname}/{self.WAV_FILE_NAME}"

        if not os.path.exists(FINAL_WAV_PATH):
            # Download the .mp4 file
            audiostream = self.yt.streams.filter(only_audio=True).first()
            outfile_path = audiostream.download(pathname)

            # Convert the .mp4 file to .wav
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


class VideoTranscription:
    """Performs transcription on a PDF or a link to a youtube video"""

    def __init__(self, datalink) -> None:
        self.datalink = datalink
        self.model = whisper.load_model('base')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
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

    def get_text_from_link(self) -> dict:

        # Get the names of the stored wav files
        YOUTUBE_VIDEO_ID = self.datalink.split("=")[1]
        FOLDER_NAME = f"./tests/{YOUTUBE_VIDEO_ID}"

        # Get the audio file
        audio_file = DownloadAudio(self.datalink)

        # Get the names of the stored wav files
        file_names = audio_file.download(FOLDER_NAME)

        # # Get the transcription of each segment
        text_transcriptions = ""
        segments = []
        for file_name in file_names:
            
            audio = open(file_name, "rb")

            # Get the transcription
            # We are guaranteed that this will be under the max size
            chunk_transcription = self.model.transcribe(file_name, fp16=False)
            text_transcriptions += chunk_transcription["text"].replace("$", "\$")
            segments.append(chunk_transcription["segments"])

        # Flatten the segments 
        segments = [segment for chunk in segments for segment in chunk]

        final_transcription = {
            "title": audio_file.get_yt_title(),
            "text": text_transcriptions,
            "segments": segments
        }

        return final_transcription
    
    
@st.cache_data
def convert_pdf_to_txt_pages(path):
    texts = []
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    size = 0
    c = 0
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    
    for page in PDFPage.get_pages(path):
        interpreter.process_page(page)
        t = retstr.getvalue()     
        if c == 0:
            texts.append(t)
        else:
            texts.append(t[size:])
        c = c + 1
        size = len(t)
        
    device.close()
    retstr.close()
    return texts, nbPages    
    
class PDFTranscription:
    
    def __init__(self, title):
        self.title = title
        self.folder_name = f"./tests/{self.title}".replace(' ', '')
        self.folder_name = self.folder_name[:self.folder_name.rindex('.')]
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def transcribe(self, pdf_file):
        # stringio = StringIO(pdf_file.getvalue().decode("ISO-8859-1"))
        # pdf_transcription = stringio.read() 
        
        # if not os.path.exists(f"{self.folder_name}"):
        #     os.mkdir(self.folder_name)
        
        # sentences = pdf_transcription.split("\n")
        # segments = []
        # for i, sentence in enumerate(sentences):
        #     segment = {
        #         "id":i,
        #         "text":sentence,
        #         "tokens":self.encoding.encode(sentence)
        #     }
            
        #     segments.append(segment)
        path = pdf_file.read()
        text, nbpages = convert_pdf_to_txt_pages(path)
        final_transcription = {
            "title":self.title,
            # "text":pdf_transcription,
            # "segments":segments,
            "pages": nbpages,
            "texts":text
        }        
        return final_transcription
        
        
        