# Function Calling Voice Assistant Guide

## Overview

The voice assistant has been enhanced with **function calling capabilities** using **Llama 3** through **Ollama**. The assistant can now intelligently detect when users need mathematical calculations or arXiv paper searches and automatically call appropriate tools to provide accurate results.

## Architecture

```
Audio Input → ASR (Whisper) → LLM (Llama 3 + Function Detection) → Tool Router → TTS (gTTS) → Audio Output
                                      ↓
                               [calculate, search_arxiv]
```

## Key Features

### 🧮 Mathematical Calculations
- Evaluates mathematical expressions using SymPy
- Supports basic arithmetic, algebra, calculus (derivatives, integrals)
- Handles symbolic mathematics and numerical approximations

### 📚 arXiv Paper Search
- Simulates academic paper searches
- Returns relevant research summaries
- Covers major research areas (AI, physics, mathematics, etc.)

### 💬 Natural Conversation
- Maintains normal conversational abilities
- Contextual memory across conversation turns
- Graceful fallback to OpenAI or rule-based responses

## Setup Instructions

### 1. Install Ollama
```bash
# Visit https://ollama.ai and install for your platform
# Or use package managers:

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull Llama 3 Model
```bash
ollama pull llama3
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional)
Create a `.env` file:
```env
# Ollama Configuration (REQUIRED)
USE_OLLAMA=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3

# General Settings
MAX_TOKENS=150
TEMPERATURE=0.7
CONVERSATION_TURNS_LIMIT=5
```

### 5. Start the Voice Assistant
```bash
python main.py
```

The server will start on `http://localhost:8000`

## Usage Examples

### Mathematical Calculations

**User:** "What is 2 plus 2?"
**Assistant:** The result is 4

**User:** "What is the derivative of x squared?"
**Assistant:** The result is 2*x

**User:** "Calculate the square root of 16"
**Assistant:** The result is 4

### Academic Research

**User:** "Find papers about quantum entanglement"
**Assistant:** arXiv search results for 'quantum entanglement': Recent research on quantum entanglement shows promising applications in quantum computing and cryptography...

**User:** "Search for machine learning research"
**Assistant:** arXiv search results for 'machine learning': Current machine learning research focuses on transformer architectures, few-shot learning, and ethical AI...

### Normal Conversation

**User:** "How are you doing today?"
**Assistant:** I'm doing wonderfully, thank you for asking! I'm here and ready to help you with any questions, calculations, or research you might need.

## Function Call Flow

### 1. User Query Processing
- Audio input is transcribed using Whisper ASR
- Text is sent to the LLM component

### 2. LLM Decision Making
Llama 3 analyzes the query and decides:
- **Mathematical query** → Outputs JSON function call for `calculate`
- **Research query** → Outputs JSON function call for `search_arxiv`
- **Conversational query** → Responds with normal text

### 3. Function Call Format
```json
{
  "function": "calculate",
  "arguments": {
    "expression": "2+2"
  }
}
```

```json
{
  "function": "search_arxiv", 
  "arguments": {
    "query": "quantum computing"
  }
}
```

### 4. Tool Execution
- `route_llm_output()` parses the JSON response
- Appropriate tool function is called
- Result is returned as the assistant's response

### 5. Text-to-Speech
- Response is converted to audio using gTTS
- Audio is played back to the user

## Sample Conversation Logs

### Mathematical Calculation
```
1. User Query: "What is the derivative of x squared?"
2. Raw LLM Response: {"function": "calculate", "arguments": {"expression": "derivative of x**2"}}
3. Function Call: calculate('derivative of x**2')
4. Tool Output: The result is 2*x
5. Final Response: "The result is 2*x"
```

### arXiv Research
```
1. User Query: "Find papers about quantum entanglement"
2. Raw LLM Response: {"function": "search_arxiv", "arguments": {"query": "quantum entanglement"}}
3. Function Call: search_arxiv('quantum entanglement')
4. Tool Output: arXiv search results for 'quantum entanglement': Recent research shows...
5. Final Response: "arXiv search results for 'quantum entanglement': Recent research shows..."
```

### Normal Conversation
```
1. User Query: "How are you doing today?"
2. Raw LLM Response: I'm doing great, thank you for asking! I'm here to help...
3. Function Call: None (normal text response)
4. Tool Output: N/A
5. Final Response: "I'm doing great, thank you for asking! I'm here to help..."
```

## API Endpoints

All existing endpoints remain unchanged:

- `GET /` - Health check
- `POST /chat/` - Main voice chat endpoint
- `GET /conversation/status` - Get conversation length  
- `POST /conversation/clear` - Clear conversation history
- `POST /set-openai-key` - Set OpenAI API key

## Testing

Run the test suite to verify functionality:

```bash
python3 test_function_calling.py
```

This will test:
- Individual tool functions
- Function call routing logic
- Sample conversation scenarios

## System Requirements

The system **requires** Llama 3 via Ollama to operate:

- **REQUIRED**: Ollama running with Llama 3 model
- **NO FALLBACKS**: System will fail if Llama 3 is unavailable
- **Pure Llama 3**: All responses generated by the model only

If Llama 3 is not available, the system will return an error and stop processing.

## Extending the System

### Adding New Tools

1. **Create the tool function** in `tools.py`:
```python
def my_new_tool(param: str) -> str:
    # Implementation
    return result
```

2. **Update the router** in `tools.py`:
```python
elif func_name == "my_new_tool":
    param = args.get("param", "")
    return my_new_tool(param)
```

3. **Update the system prompt** in `llm.py`:
```python
FUNCTION_CALLING_PROMPT = """
...
Available functions:
- calculate: For math expressions
- search_arxiv: For research queries  
- my_new_tool: For new functionality
...
"""
```

### Configuration Options

Environment variables for customization:

- `USE_OLLAMA`: Enable/disable Ollama (default: true)
- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (default: llama3)
- `MAX_TOKENS`: Response length limit (default: 150)
- `TEMPERATURE`: Response randomness (default: 0.7)

## Troubleshooting

### Common Issues

1. **Ollama not responding**
   - Ensure Ollama is running: `ollama serve`
   - Check model is available: `ollama list`

2. **Function calls not working**
   - Verify Llama 3 model is pulled: `ollama pull llama3`
   - Check system prompt is properly formatted
   - Test with `test_function_calling.py`

3. **Dependencies missing**
   - Install all requirements: `pip install -r requirements.txt`
   - Verify sympy is installed for math calculations

4. **Port conflicts**
   - Default Ollama port: 11434
   - Default voice assistant port: 8000
   - Update configuration if needed

## Performance Notes

- **First query**: May take longer due to model loading
- **Subsequent queries**: Faster due to model caching
- **Function calls**: Add ~1-2 seconds for tool execution
- **Memory usage**: Depends on Ollama model size (~4-8GB for Llama 3)

## Security Considerations

- Mathematical expressions are evaluated using SymPy (safe)
- arXiv searches are read-only operations
- No file system access or network operations in tools
- Input validation and error handling throughout

---

## 🎉 Ready to Use!

Your voice assistant now has intelligent function calling capabilities. It can seamlessly handle mathematical calculations, research queries, and normal conversation, providing a truly autonomous research assistant experience.