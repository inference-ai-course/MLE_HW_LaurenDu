import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
CONVERSATION_TURNS_LIMIT = int(os.getenv("CONVERSATION_TURNS_LIMIT", "5"))

try:
    from openai import OpenAI
    
    # Initialize OpenAI client
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"[INFO] OpenAI client initialized with model: {OPENAI_MODEL}")
        USE_OPENAI = True
    else:
        print("[WARNING] No OpenAI API key provided. Will use fallback responses.")
        client = None
        USE_OPENAI = False
        
except ImportError as e:
    print(f"[ERROR] Failed to import OpenAI: {e}")
    print("[INFO] Using rule-based responses as fallback...")
    client = None
    USE_OPENAI = False

# Initialize conversation history with system prompt
conversation_history = [
    {"role": "system", "content": "You are a helpful and friendly voice assistant. Respond conversationally in 2-3 sentences. Be warm and engaging while staying concise since this is a voice conversation."}
]

def generate_response(user_text):
    """
    Generate a response using OpenAI API with conversation memory
    
    Args:
        user_text (str): User's transcribed text input
        
    Returns:
        str: Generated response text
    """
    global conversation_history
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_text})
    
    # Keep only last 5 turns (system + 10 messages: 5 user + 5 assistant)
    if len(conversation_history) > (CONVERSATION_TURNS_LIMIT * 2 + 1):  # +1 for system message
        # Keep system message and last 10 messages
        conversation_history = [conversation_history[0]] + conversation_history[-(CONVERSATION_TURNS_LIMIT * 2):]
    
    try:
        if USE_OPENAI and client:
            # Generate response using OpenAI API
            print("[INFO] Generating response with OpenAI...")
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=conversation_history,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            bot_response = response.choices[0].message.content.strip()
            print(f"[INFO] OpenAI response: {bot_response}")
            
        else:
            # Fallback to intelligent rule-based responses
            print("[INFO] Using enhanced rule-based response...")
            bot_response = generate_fallback_response(user_text)
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": bot_response})
        
        return bot_response
        
    except Exception as e:
        print(f"[ERROR] Response generation failed: {str(e)}")
        # Ultimate fallback response
        fallback_response = "I'm sorry, I'm having trouble processing that right now. Could you please try again?"
        conversation_history.append({"role": "assistant", "content": fallback_response})
        return fallback_response

def generate_fallback_response(user_text):
    """
    Generate intelligent fallback responses when OpenAI is not available
    """
    import random
    
    user_lower = user_text.lower()
    
    # Context-aware responses based on keywords
    if any(word in user_lower for word in ['hello', 'hi', 'hey']):
        responses = [
            "Hello there! It's great to hear from you. What's on your mind today?",
            "Hi! I'm glad you're here. How can I help you?",
            "Hey! Nice to chat with you. What would you like to talk about?"
        ]
    elif any(word in user_lower for word in ['how are you', 'how do you', 'what are you']):
        responses = [
            "I'm doing well, thank you for asking! I'm here to chat and help however I can. How are you doing?",
            "I'm great! I enjoy our conversations. What's going well in your day?",
            "I'm doing wonderfully! I love learning about what matters to you. How has your day been?"
        ]
    elif any(word in user_lower for word in ['thanks', 'thank you']):
        responses = [
            "You're very welcome! I'm happy I could help. Is there anything else you'd like to discuss?",
            "My pleasure! I really enjoy our conversation. What else is on your mind?",
            "Absolutely! I'm glad to be helpful. Feel free to ask me anything else."
        ]
    elif any(word in user_lower for word in ['bye', 'goodbye', 'see you']):
        responses = [
            "Goodbye! It was wonderful talking with you. Take care and have a great day!",
            "See you later! Thanks for the lovely conversation. Hope to chat again soon!",
            "Farewell! I really enjoyed our time together. Wishing you all the best!"
        ]
    elif '?' in user_text:
        responses = [
            f"That's a thoughtful question about {user_text.split()[0] if user_text.split() else 'that topic'}. Let me think about that with you.",
            "I appreciate you asking! That's something worth exploring together.",
            "Great question! I'd love to help you think through that."
        ]
    else:
        # Check conversation context for more intelligent responses
        recent_topics = []
        for msg in conversation_history[-3:]:
            if msg["role"] == "user":
                recent_topics.extend(msg["content"].lower().split())
        
        if any(word in recent_topics for word in ['feel', 'feeling', 'emotion', 'mood']):
            responses = [
                "It sounds like you're reflecting on some important feelings. I'm here to listen.",
                "Emotions can be complex. Thank you for sharing what's on your heart.",
                "I appreciate you opening up about how you're feeling. Tell me more if you'd like."
            ]
        elif any(word in recent_topics for word in ['work', 'job', 'career', 'office']):
            responses = [
                "Work life can definitely be a lot to navigate. What aspects are you thinking about most?",
                "Career topics are so important. I'd love to hear more about your perspective.",
                "Work situations can be challenging. What's been on your mind about it?"
            ]
        else:
            responses = [
                f"That's really interesting that you mentioned {user_text.split()[0] if user_text.split() else 'that'}. I'd love to hear more about your thoughts.",
                "I find that fascinating! Can you tell me more about what you're thinking?",
                "That's a great point you've brought up. What's your perspective on it?",
                "Thanks for sharing that with me. I'm curious to learn more about your experience.",
                "I appreciate you telling me about this. What's been most important to you about it?"
            ]
    
    return random.choice(responses)

def clear_conversation():
    """Clear the conversation history"""
    global conversation_history
    conversation_history = [
        {"role": "system", "content": "You are a helpful and friendly voice assistant. Respond conversationally in 2-3 sentences. Be warm and engaging while staying concise since this is a voice conversation."}
    ]
    print("[INFO] Conversation history cleared.")

def get_conversation_length():
    """Get the current conversation length (excluding system message)"""
    return len(conversation_history) - 1

def set_openai_key(api_key):
    """Set the OpenAI API key dynamically"""
    global client, USE_OPENAI
    try:
        client = OpenAI(api_key=api_key)
        USE_OPENAI = True
        print("[INFO] OpenAI API key updated successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to set OpenAI API key: {e}")
        return False