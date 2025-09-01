from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import logging
from asr import transcribe_audio
from llm import generate_response, clear_conversation, get_conversation_length
from tts import synthesize_speech, cleanup_audio_file

app = FastAPI(title="Voice Chatbot", description="Real-time voice chatbot with ASR, LLM, and TTS")

# Add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Voice chatbot is running"}

@app.get("/conversation/status")
async def conversation_status():
    return {"conversation_length": get_conversation_length()}

@app.post("/conversation/clear")
async def clear_conversation_endpoint():
    clear_conversation()
    return {"message": "Conversation cleared"}

# Note: OpenAI support removed - now using Llama 3 via Ollama only

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_path = None
    try:
        logger.info(f"Received audio file: {file.filename}, content_type: {file.content_type}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Validate file size (max 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Read audio bytes
        audio_bytes = await file.read()
        logger.info(f"Read {len(audio_bytes)} bytes of audio data")
        
        # Step 1: ASR - Transcribe audio to text
        logger.info("Starting ASR transcription...")
        user_text = transcribe_audio(audio_bytes)
        logger.info(f"Transcribed text: {user_text}")
        
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio file")
        
        # Step 2: LLM - Generate response using Llama 3
        logger.info("Generating LLM response with Llama 3...")
        bot_text = generate_response(user_text)
        logger.info(f"Generated response: {bot_text}")
        
        # Step 3: TTS - Convert response to speech
        logger.info("Converting response to speech...")
        audio_path = synthesize_speech(bot_text)
        logger.info(f"Generated audio file: {audio_path}")
        
        # Return audio response
        return FileResponse(
            audio_path, 
            media_type="audio/wav", 
            filename="response.wav",
            background=lambda: cleanup_audio_file(audio_path)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        if audio_path:
            cleanup_audio_file(audio_path)
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        if audio_path:
            cleanup_audio_file(audio_path)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)