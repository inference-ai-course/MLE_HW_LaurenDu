#!/usr/bin/env python3
"""
NLP Conference Talk URLs for Whisper Transcription Bot
Curated list of short NLP conference presentations and talks.
"""

# Curated NLP-related YouTube videos for transcription
# These are educational/conference-style videos that are ~3-5 minutes each
NLP_CONFERENCE_URLS = [
    # Stanford NLP Talks and Tutorials (Short segments)
    "https://www.youtube.com/watch?v=kEMJRjEdNzM",  # "What is NLP?" - Stanford
    "https://www.youtube.com/watch?v=fOvTtapxa9c",  # "Introduction to Transformers" - Short explanation
    "https://www.youtube.com/watch?v=S27pHKBEp30",  # "BERT Explained" - Short version
    
    # AI/ML Conference Short Presentations
    "https://www.youtube.com/watch?v=LWiM-LuRe6w",  # "Attention Mechanism Explained" - 3 mins
    "https://www.youtube.com/watch?v=TQQlZhbC5ps",  # "Word Embeddings in 3 Minutes"
    "https://www.youtube.com/watch?v=aircAruvnKk",  # "Neural Networks Explained" - 3Blue1Brown
    
    # Academic NLP Content (Short form)
    "https://www.youtube.com/watch?v=ySEx_Bqxvvo",  # "Language Models" - MIT
    "https://www.youtube.com/watch?v=OQQ-W_63UgQ",  # "Machine Translation" - Stanford
    "https://www.youtube.com/watch?v=WCUNPb-5EYI",  # "Named Entity Recognition" 
    
    # Recent NLP Developments (Short explanations)
    "https://www.youtube.com/watch?v=TrWqRMJZU8A",  # "GPT Explained in 3 Minutes"
]

# Alternative shorter videos if the above are too long
BACKUP_URLS = [
    "https://www.youtube.com/watch?v=SZorAJ4I-sA",  # "RNN vs LSTM vs GRU" - 4 mins
    "https://www.youtube.com/watch?v=fjJOgb-E41w",  # "Transformer Architecture" - Short
    "https://www.youtube.com/watch?v=t0P_R0NRrEI",  # "Self-Attention Mechanism"
]

def get_curated_urls(max_videos: int = 10) -> list:
    """Get curated list of NLP video URLs."""
    all_urls = NLP_CONFERENCE_URLS + BACKUP_URLS
    return all_urls[:max_videos]

def get_test_urls(count: int = 3) -> list:
    """Get a few URLs for quick testing."""
    return NLP_CONFERENCE_URLS[:count]

def get_all_urls() -> list:
    """Get all available URLs."""
    return NLP_CONFERENCE_URLS + BACKUP_URLS

if __name__ == "__main__":
    urls = get_curated_urls()
    print(f"Found {len(urls)} NLP video URLs:")
    for i, url in enumerate(urls, 1):
        print(f"{i:2}. {url}") 