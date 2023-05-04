# For downloading from youtube and transcribing audio
from pytube import YouTube
from moviepy.editor import * 
from pydub import AudioSegment
from pydub.utils import make_chunks
import pydub
from yt_dlp import YoutubeDL
from pathlib import Path
import subprocess

# For getting text from PDF
from zipfile import ZipFile
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

# For transcription
import openai, whisper, torch
from faster_whisper import WhisperModel
import tiktoken
from nltk import tokenize

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
MODEL_SIZE = "base"

class DownloadAudio:
    """Downloads the audio from a youtube video and saves it to multiple .wav files in the specified folder"""

    def __init__(self, link) -> None:
        self.link = link
        with YoutubeDL() as ydl:
            self.yt = ydl.extract_info(self.link, download=False)
            
        self.YOUTUBE_VIDEO_ID = link.split("=")[1]
        self.WAV_FILE_NAME = f"{self.YOUTUBE_VIDEO_ID}.wav"

    def get_yt_title(self) -> str:
        """Returns the title of the youtube video"""
        return self.yt["title"]

    def download(self, pathname:str) -> str:
        """
        Download the audio from the youtube video and saves it to multiple .wav files
        in the specified folder. Returns a list of the paths to the .wav files.
        """

        # Check if the folder for the VIDEO_ID exists
        if not os.path.exists(pathname):
            os.mkdir(pathname)
        FINAL_WAV_PATH = f"{pathname}/{self.WAV_FILE_NAME}"

        if not os.path.exists(FINAL_WAV_PATH):
            print("\n\n\n DOWNLOADING AUDIO \n\n\n")
            current_dir = os.getcwd()
            print(current_dir)
            executable_path = os.path.join(current_dir, "exec/yt-dlp_linux")
            
            # Download the video as an audio file using youtube-dl
            original_download_path = f"{pathname}/audio.wav"
            result = subprocess.run([executable_path, "-x", "--audio-format", "wav", "-o", original_download_path, self.link])
            if result.returncode != 0:
                print("Failed to download audio. Retrying...")
                return "FAILED"

            sound = AudioSegment.from_wav(original_download_path)
            sound.set_channels(1)
            sound = sound.set_frame_rate(16000)                
            sound = sound.set_channels(1)    
            sound.export(FINAL_WAV_PATH, format="wav")
            os.remove(original_download_path)
            
        # Load the input .wav file
        audio = AudioSegment.from_wav(FINAL_WAV_PATH)
    
        # Get the duration of the input file in milliseconds
        total_byte_size = os.path.getsize(FINAL_WAV_PATH)
        
        # If the total duration is less than the duration of each segment,
        # then just return the original file
        if total_byte_size < MAX_FILE_SIZE_BYTES:
            return FINAL_WAV_PATH

        # # Get the size of the wav file
        # channels = audio.channels
        # sample_width = audio.sample_width
        # duration_in_sec = math.ceil(len(audio) / 1000)
        # sample_rate = audio.frame_rate
        # bit_rate = sample_width * 8
        # wav_file_size = (sample_rate * bit_rate * channels * duration_in_sec) / 8

        # # Get the length of each chunk in milliseconds and make the chunks
        # chunk_length_in_sec = math.ceil((duration_in_sec * MAX_FILE_SIZE_BYTES ) / wav_file_size)   #in sec
        # chunk_length_ms = chunk_length_in_sec * 1000
        # chunks = make_chunks(audio, chunk_length_ms)

        # # Export all of the individual chunks as wav files
        # chunk_names = []
        # for i, chunk in enumerate(chunks):
        #     chunk_name = f"{self.YOUTUBE_VIDEO_ID}_{i}.wav"
        #     output_chunk_path = f"{pathname}/{chunk_name}"
        #     chunk_names.append(output_chunk_path)
        #     chunk.export(f"{output_chunk_path}", format="wav")
        
        return FINAL_WAV_PATH


class VideoTranscription:
    """Performs transcription on a PDF or a link to a youtube video"""

    def __init__(self, datalink) -> None:
        self.datalink = datalink
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
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
        original_file_name = audio_file.download(FOLDER_NAME)
        print(original_file_name)
        # Get the transcription of each audio chunk
        text_transcriptions = ""
        # for file_name in file_names:
        # Get the transcription
        chunk_segments, _ = self.model.transcribe(original_file_name, beam_size=5)
        for chunk_segment in chunk_segments:
            text_transcriptions += chunk_segment.text.replace("$", "\$")    

        # Tokenize each sentence of the transcription. 
        sentences = tokenize.sent_tokenize(text_transcriptions)
        segments = []
        for i, sentence in enumerate(sentences):
            segment = {
                "id":i,
                "text":sentence,
                "tokens":self.encoding.encode(sentence)
            }
            segments.append(segment)
        
        final_transcription = {
            "title": audio_file.get_yt_title(),
            "text": text_transcriptions,
            "segments": segments
        }

        return final_transcription


class AudioTranscription:
    """Performs transcription on a MP3 file"""

    def __init__(self, audio_file) -> None:
        self.file = audio_file
        self.title = self.file.name
        self.folder_name = f"./tests/{self.title}".replace(' ', '')
        self.folder_name = self.folder_name[:self.folder_name.rindex('.')]
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        
    def get_redacted_name(self):
        return self.folder_name
        
    def transcribe(self) -> dict:
        """Returns the transcription of the MP3 audio as a string"""

        start_time = time.time()
        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)
        
        if self.title.endswith('wav'):
            audio = pydub.AudioSegment.from_wav(self.file)
            file_type = 'wav'
        elif self.title.endswith('mp3'):
            audio = pydub.AudioSegment.from_mp3(self.file)
            file_type = 'mp3'

        save_path = Path(self.folder_name) / self.file.name
        audio.export(save_path, format=file_type)
        final_wav_path = save_path
        
        if file_type == 'mp3':
            sound = AudioSegment.from_mp3(save_path)
            final_wav_path = self.folder_name + "/" +  self.title[:-4]+'.wav'
            sound.export(final_wav_path, format="wav")
        
        chunk_segments, info = self.model.transcribe(final_wav_path, beam_size=5)
        text_transcriptions = ""
        for chunk_segment in chunk_segments:
            text_transcriptions += chunk_segment.text.replace("$", "\$")    

        # Tokenize each sentence of the transcription. 
        sentences = tokenize.sent_tokenize(text_transcriptions)
        segments = []
        for i, sentence in enumerate(sentences):
            segment = {
                "id":i,
                "text":sentence,
                "tokens":self.encoding.encode(sentence)
            }
            segments.append(segment)
        
        final_transcription = {
            "title": self.title,
            "text": text_transcriptions,
            "segments": segments
        }
        end_time = time.time()
        print(f"transcription took {end_time - start_time} seconds")

        return final_transcription

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
    
    def __init__(self, pdf_file):
        self.file = pdf_file
        self.title = pdf_file.name
        self.folder_name = f"./tests/{self.title}".replace(' ', '')
        self.folder_name = self.folder_name[:self.folder_name.rindex('.')]
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def get_redacted_name(self):
        return self.folder_name
    
    def transcribe(self):
        text, nbpages = convert_pdf_to_txt_pages(self.file)
        pdf_transcription = ''.join(text)
        
        sentences = tokenize.sent_tokenize(pdf_transcription)
        segments = []
        for i, sentence in enumerate(sentences):
            segment = {
                "id":i,
                "text":sentence,
                "tokens":self.encoding.encode(sentence)
            }
            
            segments.append(segment)
        
        final_transcription = {
            "title":self.title,
            "text":pdf_transcription,
            "segments":segments
        }        
        return final_transcription
        
        
        