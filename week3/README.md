# Real-Time Voice Chatbot

A complete voice chatbot implementation with ASR (Automatic Speech Recognition), LLM response generation, and TTS (Text-to-Speech) capabilities.

## Features

- 🎤 **Audio Input**: Accept audio files via HTTP POST
- 🔤 **Speech Recognition**: Transcribe audio to text using OpenAI Whisper
- 🧠 **LLM Response**: Generate contextual responses with conversation memory (5 turns)
- 🔊 **Text-to-Speech**: Convert responses back to natural speech
- 💬 **Conversation Memory**: Maintains context across multiple interactions
- 🌐 **Web Interface**: Simple HTML test client for easy testing

## Architecture

```
Audio Input → ASR (Whisper) → LLM (Transformers) → TTS (gTTS) → Audio Output
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Additional System Dependencies** (optional, for better audio support):
   ```bash
   # On macOS
   brew install ffmpeg
   
   # On Ubuntu/Debian
   sudo apt update
   sudo apt install ffmpeg
   ```

## Usage

### Starting the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

### API Endpoints

- **POST `/chat/`**: Main chat endpoint - upload audio, get audio response
- **GET `/`**: Health check
- **GET `/conversation/status`**: Get current conversation length
- **POST `/conversation/clear`**: Clear conversation history

### Testing with the Web Client

1. Open `test_client.html` in your web browser
2. Either:
   - Record audio directly using your microphone
   - Upload an existing audio file
3. The chatbot will process your audio and respond with speech

### API Usage Example

```python
import requests

# Send audio file to chatbot
with open('your_audio.wav', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/chat/', files=files)
    
    # Save response audio
    with open('response.wav', 'wb') as out:
        out.write(response.content)
```

## Configuration

### Model Configuration

You can modify the models used in each component:

- **ASR Model** (`asr.py`): Change Whisper model size (tiny, small, medium, large)
- **LLM Model** (`llm.py`): Switch to different language models (currently using DialoGPT-medium)
- **TTS Language** (`tts.py`): Change language for speech synthesis

### Memory Management

The chatbot maintains conversation memory for 5 turns (10 messages total). You can:
- Check conversation status: `GET /conversation/status`
- Clear conversation: `POST /conversation/clear`

## File Structure

```
├── main.py              # FastAPI server and main endpoints
├── asr.py               # Speech recognition module
├── llm.py               # Language model and conversation memory
├── tts.py               # Text-to-speech module
├── requirements.txt     # Python dependencies
├── test_client.html     # Web-based test client
└── README.md           # This file
```

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Microphone access (for recording in web client)

## Performance Notes

- **First Request**: May take longer due to model loading
- **GPU**: Significantly faster with CUDA-enabled PyTorch
- **Model Size**: Smaller models = faster response, larger models = better quality
- **Memory**: Ensure sufficient RAM for model loading (2-8GB depending on models)

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: Install PyTorch with CUDA support if using GPU
3. **Audio Issues**: Check file format compatibility (WAV, MP3, etc.)
4. **Memory Errors**: Try smaller models or increase system RAM
5. **Microphone Access**: Ensure browser has microphone permissions

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.