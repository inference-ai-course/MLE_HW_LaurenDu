import whisper
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load Whisper model (configurable via environment variable)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
asr_model = whisper.load_model(WHISPER_MODEL)

def transcribe_audio(audio_bytes):
    """
    Transcribe audio bytes to text using OpenAI Whisper
    
    Args:
        audio_bytes (bytes): Raw audio file bytes
        
    Returns:
        str: Transcribed text
    """
    # Create temporary file to save audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name
    
    try:
        # Transcribe audio using Whisper
        result = asr_model.transcribe(temp_path)
        transcribed_text = result["text"].strip()
        
        return transcribed_text
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)