import os
import json
from dotenv import load_dotenv
from tools import route_llm_output, calculate, search_arxiv
import re

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
CONVERSATION_TURNS_LIMIT = int(os.getenv("CONVERSATION_TURNS_LIMIT", "5"))

# Ollama configuration - using Llama 3 as primary LLM
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # Llama 3 model
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"

# Initialize Ollama client with Llama 3
ollama_client = None
USE_OLLAMA_CLIENT = False

if USE_OLLAMA:
    try:
        import ollama
        ollama_client = ollama.Client(host=OLLAMA_HOST)
        # Test connection by listing models
        models = ollama_client.list()
        available_models = [model.model for model in models.models]
        
        # Check if llama3 model is available (might be llama3:latest)
        model_found = any(OLLAMA_MODEL in model for model in available_models)
        if model_found:
            USE_OLLAMA_CLIENT = True
            print(f"[INFO] Ollama client initialized with Llama 3 model: {OLLAMA_MODEL}")
        else:
            print(f"[ERROR] Llama 3 model {OLLAMA_MODEL} not found. Available models: {available_models}")
            print("[ERROR] Please install Llama 3: ollama pull llama3")
            USE_OLLAMA_CLIENT = False
    except ImportError as e:
        print(f"[ERROR] Ollama not available: {e}")
        print("[ERROR] Install ollama: pip install ollama")
        USE_OLLAMA_CLIENT = False
    except Exception as e:
        print(f"[ERROR] Failed to connect to Ollama: {e}")
        print("[ERROR] Ensure Ollama is running: ollama serve")
        USE_OLLAMA_CLIENT = False
else:
    print("[ERROR] Ollama disabled via configuration - system requires Llama 3")

# Function calling system prompt for Ollama
FUNCTION_CALLING_PROMPT = """
You are a helpful and friendly voice assistant with access to special tools. Respond conversationally in 2-3 sentences and be warm and engaging since this is a voice conversation.

When users ask questions that require:
1. Mathematical calculations or expressions (like "what is 2+2", "derivative of x squared", "solve x^2+5x+6")
2. Academic research or arXiv paper searches (like "find papers about quantum computing", "research on neural networks")

You should respond with a JSON function call in this exact format:
{"function": "function_name", "arguments": {"parameter": "value"}}

Available functions:
- calculate: For math expressions. Use {"function": "calculate", "arguments": {"expression": "math_expression"}}
- search_arxiv: For research queries. Use {"function": "search_arxiv", "arguments": {"query": "search_terms"}}

For all other conversational queries, respond normally with friendly text (no JSON).

Examples:
User: "What is 2 plus 2?"
Assistant: {"function": "calculate", "arguments": {"expression": "2+2"}}

User: "Find papers about machine learning"
Assistant: {"function": "search_arxiv", "arguments": {"query": "machine learning"}}

User: "How are you doing?"
Assistant: I'm doing great, thank you for asking! I'm here to help with any questions or tasks you might have.
"""

# Standard conversation prompt for OpenAI and fallback
STANDARD_PROMPT = "You are a helpful and friendly voice assistant. Respond conversationally in 2-3 sentences. Be warm and engaging while staying concise since this is a voice conversation."

# Initialize conversation history with system prompt
conversation_history = [
    {"role": "system", "content": STANDARD_PROMPT}
]

def extract_and_execute_function_calls(response_text):
    """
    Extract JSON function calls from Llama 3 response and execute them
    
    Args:
        response_text (str): Raw response from Llama 3
        
    Returns:
        str: Either the function result or the original text if no function call
    """
    try:
        # Look for JSON function call pattern
        json_pattern = r'\{"function":\s*"([^"]+)",\s*"arguments":\s*\{[^}]+\}\}'
        match = re.search(json_pattern, response_text)
        
        if match:
            function_call_json = match.group(0)
            print(f"[INFO] Function call detected: {function_call_json}")
            
            # Parse and execute the function call
            parsed = json.loads(function_call_json)
            func_name = parsed.get("function")
            args = parsed.get("arguments", {})
            
            if func_name == "calculate":
                expr = args.get("expression", "")
                result = calculate(expr)
                print(f"[INFO] Executed calculate('{expr}') -> {result}")
                return result
            elif func_name == "search_arxiv":
                query = args.get("query", "")
                result = search_arxiv(query)
                print(f"[INFO] Executed search_arxiv('{query}') -> {result}")
                return result
            else:
                print(f"[WARNING] Unknown function: {func_name}")
                return response_text
        else:
            # No function call found, return original response
            return response_text
            
    except Exception as e:
        print(f"[ERROR] Function call extraction failed: {e}")
        return response_text

def generate_ollama_response(user_text):
    """
    Generate a response using Ollama with function calling support
    
    Args:
        user_text (str): User's transcribed text input
        
    Returns:
        str: Generated response text (may be processed through tools)
    """
    try:
        # Create conversation context for Ollama
        conversation_context = f"{FUNCTION_CALLING_PROMPT}\n\nConversation so far:\n"
        
        # Add recent conversation history (last few turns)
        recent_history = conversation_history[-6:]  # Last 3 turns (user + assistant pairs)
        for msg in recent_history:
            if msg["role"] == "user":
                conversation_context += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_context += f"Assistant: {msg['content']}\n"
        
        conversation_context += f"User: {user_text}\nAssistant: "
        
        print("[INFO] Generating response with Ollama...")
        
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=conversation_context,
            options={
                "temperature": TEMPERATURE,
                "top_p": 0.9,
                "max_tokens": MAX_TOKENS * 2,  # Allow more tokens for function calls
            }
        )
        
        raw_response = response['response'].strip()
        print(f"[INFO] Ollama raw response: {raw_response}")
        
        # Extract and execute function calls from the response
        processed_response = extract_and_execute_function_calls(raw_response)
        print(f"[INFO] Processed response: {processed_response}")
        
        return processed_response
        
    except Exception as e:
        print(f"[ERROR] Ollama response generation failed: {str(e)}")
        raise e

def generate_response(user_text):
    """
    Generate a response using Llama 3 with conversation memory
    Requires Ollama with Llama 3 model to be running
    
    Args:
        user_text (str): User's transcribed text input
        
    Returns:
        str: Generated response text
        
    Raises:
        Exception: If Llama 3 is not available or fails to respond
    """
    global conversation_history
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_text})
    
    # Keep only last 5 turns (system + 10 messages: 5 user + 5 assistant)
    if len(conversation_history) > (CONVERSATION_TURNS_LIMIT * 2 + 1):  # +1 for system message
        # Keep system message and last 10 messages
        conversation_history = [conversation_history[0]] + conversation_history[-(CONVERSATION_TURNS_LIMIT * 2):]
    
    # Generate response using Ollama with Llama 3 (required)
    if not USE_OLLAMA_CLIENT or not ollama_client:
        raise Exception("Llama 3 via Ollama is not available. Please ensure Ollama is running with llama3 model loaded.")
    
    try:
        bot_response = generate_ollama_response(user_text)
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": bot_response})
        
        return bot_response
        
    except Exception as e:
        print(f"[ERROR] Llama 3 via Ollama failed: {e}")
        # Remove the user message from history since we couldn't respond
        conversation_history.pop()
        raise Exception(f"Failed to generate response with Llama 3: {str(e)}")

# Rule-based fallback removed - system now requires Llama 3 via Ollama

def clear_conversation():
    """Clear the conversation history"""
    global conversation_history
    conversation_history = [
        {"role": "system", "content": STANDARD_PROMPT}
    ]
    print("[INFO] Conversation history cleared.")

def get_conversation_length():
    """Get the current conversation length (excluding system message)"""
    return len(conversation_history) - 1

def set_openai_key(api_key):
    """Legacy function - OpenAI support removed"""
    print("[INFO] OpenAI support has been removed. Using Llama 3 via Ollama only.")
    return False