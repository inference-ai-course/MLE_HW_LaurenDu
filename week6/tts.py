from gtts import gTTS
import tempfile
import os
from io import BytesIO

def synthesize_speech(text, language='en'):
    """
    Convert text to speech using Google Text-to-Speech
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code (default: 'en')
        
    Returns:
        str: Path to the generated audio file
    """
    try:
        # Create TTS object
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Create temporary file for audio output
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Save audio to temporary file
        tts.save(temp_path)
        
        return temp_path
        
    except Exception as e:
        # Create a fallback silent audio file if TTS fails
        return create_fallback_audio()

def create_fallback_audio():
    """
    Create a fallback audio file with error message
    
    Returns:
        str: Path to fallback audio file
    """
    try:
        fallback_text = "I'm sorry, there was an error generating the audio response."
        tts = gTTS(text=fallback_text, lang='en', slow=False)
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        tts.save(temp_path)
        return temp_path
        
    except Exception:
        # If even fallback fails, create empty temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        return temp_path

def cleanup_audio_file(file_path):
    """
    Clean up temporary audio file
    
    Args:
        file_path (str): Path to audio file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass  # Ignore cleanup errors