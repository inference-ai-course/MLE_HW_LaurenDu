#!/usr/bin/env python3
"""
Whisper Transcription Bot for NLP Conference Talks
Fetches YouTube audio, transcribes with Whisper, and extracts OCR text from frames.
Outputs timestamped transcripts in JSONL format.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import logging

# Import NLP video URLs
try:
    from nlp_video_urls import get_curated_urls, get_test_urls, get_all_urls
except ImportError:
    print("Warning: nlp_video_urls.py not found. Using example URLs.")
    get_curated_urls = lambda n=10: [f"https://www.youtube.com/watch?v=example{i}" for i in range(1, n+1)]
    get_test_urls = lambda n=3: [f"https://www.youtube.com/watch?v=example{i}" for i in range(1, n+1)]
    get_all_urls = lambda: [f"https://www.youtube.com/watch?v=example{i}" for i in range(1, 11)]

try:
    import whisper
    import cv2
    import pytesseract
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install openai-whisper opencv-python pytesseract pillow numpy")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhisperTranscriptionBot:
    def __init__(self, model_size: str = "base", output_file: str = "talks_transcripts.jsonl"):
        """
        Initialize the transcription bot.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            output_file: Output JSONL file path
        """
        self.model_size = model_size
        self.output_file = output_file
        self.whisper_model = None
        
    def load_whisper_model(self):
        """Load Whisper model for transcription."""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
        return self.whisper_model
    
    def download_youtube_audio(self, youtube_url: str, output_dir: str) -> Optional[str]:
        """
        Download audio from YouTube video using yt-dlp.
        
        Args:
            youtube_url: YouTube video URL
            output_dir: Directory to save audio file
            
        Returns:
            Path to downloaded audio file or None if failed
        """
        try:
            audio_file = os.path.join(output_dir, "%(title)s.%(ext)s")
            cmd = [
                "python3", "-m", "yt_dlp",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--output", audio_file,
                youtube_url
            ]
            
            logger.info(f"Downloading audio from: {youtube_url}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find the downloaded file
            for file in os.listdir(output_dir):
                if file.endswith('.wav'):
                    return os.path.join(output_dir, file)
                    
            return None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download audio: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            return None
    
    def extract_video_frames(self, youtube_url: str, output_dir: str, interval: int = 30) -> List[str]:
        """
        Extract frames from YouTube video for OCR processing.
        
        Args:
            youtube_url: YouTube video URL
            output_dir: Directory to save frames
            interval: Extract frame every N seconds
            
        Returns:
            List of frame file paths
        """
        try:
            # Download video for frame extraction
            video_file = os.path.join(output_dir, "temp_video.%(ext)s")
            cmd = [
                "python3", "-m", "yt_dlp",
                "--format", "best[height<=720]",
                "--output", video_file,
                youtube_url
            ]
            
            logger.info("Downloading video for frame extraction...")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find the downloaded video file
            video_path = None
            for file in os.listdir(output_dir):
                if file.startswith('temp_video.') and not file.endswith('.wav'):
                    video_path = os.path.join(output_dir, file)
                    break
            
            if not video_path:
                logger.warning("No video file found for frame extraction")
                return []
            
            # Extract frames using OpenCV
            frames = []
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval)
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame_path)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            # Clean up video file
            os.remove(video_path)
            
            logger.info(f"Extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def extract_text_from_frame(self, frame_path: str) -> str:
        """
        Extract text from a video frame using Tesseract OCR.
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Extracted text
        """
        try:
            # Load and preprocess image
            image = cv2.imread(frame_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing for better OCR
            # Increase contrast
            alpha = 1.5  # Contrast control
            beta = 0     # Brightness control
            enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(denoised, config='--psm 6')
            
            # Clean up extracted text
            text = ' '.join(text.split())
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from {frame_path}: {str(e)}")
            return ""
    
    def transcribe_audio(self, audio_file: str) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcription result with timestamps
        """
        try:
            model = self.load_whisper_model()
            logger.info(f"Transcribing audio: {os.path.basename(audio_file)}")
            
            result = model.transcribe(
                audio_file,
                word_timestamps=True,
                verbose=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {}
    
    def process_video(self, youtube_url: str, video_title: str = None) -> Dict[str, Any]:
        """
        Process a single YouTube video: download, transcribe, and extract OCR text.
        
        Args:
            youtube_url: YouTube video URL
            video_title: Optional video title (will be extracted if not provided)
            
        Returns:
            Processed video data
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download audio
            audio_file = self.download_youtube_audio(youtube_url, temp_dir)
            if not audio_file:
                logger.error(f"Failed to download audio for {youtube_url}")
                return {}
            
            # Transcribe audio
            transcription_result = self.transcribe_audio(audio_file)
            if not transcription_result:
                logger.error(f"Failed to transcribe {youtube_url}")
                return {}
            
            # Extract frames and OCR text
            frames = self.extract_video_frames(youtube_url, temp_dir)
            ocr_texts = []
            
            for frame_path in frames:
                text = self.extract_text_from_frame(frame_path)
                if text:
                    # Get frame timestamp based on filename
                    frame_num = int(os.path.basename(frame_path).split('_')[1].split('.')[0])
                    timestamp = frame_num * 30  # 30 second intervals
                    ocr_texts.append({
                        "timestamp": timestamp,
                        "text": text
                    })
            
            # Get video title if not provided
            if not video_title:
                video_title = transcription_result.get('text', 'Unknown Title')[:50]
            
            # Compile results
            result = {
                "url": youtube_url,
                "title": video_title,
                "duration": transcription_result.get('duration', 0),
                "language": transcription_result.get('language', 'unknown'),
                "transcript": {
                    "full_text": transcription_result.get('text', ''),
                    "segments": transcription_result.get('segments', [])
                },
                "ocr_texts": ocr_texts,
                "timestamp": int(os.path.getmtime(audio_file))
            }
            
            return result
    
    def process_multiple_videos(self, youtube_urls: List[str]) -> None:
        """
        Process multiple YouTube videos and save results to JSONL file.
        
        Args:
            youtube_urls: List of YouTube video URLs
        """
        results = []
        
        for i, url in enumerate(youtube_urls, 1):
            logger.info(f"Processing video {i}/{len(youtube_urls)}: {url}")
            
            try:
                result = self.process_video(url)
                if result:
                    results.append(result)
                    
                    # Save incrementally
                    with open(self.output_file, 'w') as f:
                        for r in results:
                            f.write(json.dumps(r) + '\n')
                    
                    logger.info(f"Processed and saved video {i}")
                else:
                    logger.error(f"Failed to process video {i}: {url}")
                    
            except Exception as e:
                logger.error(f"Error processing video {i} ({url}): {str(e)}")
                continue
        
        logger.info(f"Completed processing {len(results)} videos. Results saved to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Whisper Transcription Bot for NLP Conference Talks")
    parser.add_argument("--urls", nargs="+", help="YouTube URLs to process")
    parser.add_argument("--urls-file", help="File containing YouTube URLs (one per line)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--output", default="talks_transcripts.jsonl", help="Output JSONL file")
    
    args = parser.parse_args()
    
    # Get URLs from command line or file
    urls = []
    if args.urls:
        urls = args.urls
    elif args.urls_file:
        try:
            with open(args.urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"URLs file not found: {args.urls_file}")
            return
    else:
        # Use curated NLP conference talk URLs
        urls = get_curated_urls(10)
        logger.info(f"Using {len(urls)} curated NLP conference talk URLs")
    
    if not urls:
        logger.error("No URLs provided")
        return
    
    # Initialize bot and process videos
    bot = WhisperTranscriptionBot(model_size=args.model, output_file=args.output)
    bot.process_multiple_videos(urls)

if __name__ == "__main__":
    main()