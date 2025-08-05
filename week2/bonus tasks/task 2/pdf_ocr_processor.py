#!/usr/bin/env python3
"""
Batch OCR for arXiv PDFs
Processes PDFs from arxiv_clean.json using Tesseract OCR with layout preservation.
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path

# Core dependencies
import requests
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivPDFOCR:
    """Batch OCR processor for arXiv PDFs with layout preservation."""
    
    def __init__(self, output_dir: str = "pdf_ocr"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.pdfs_dir = self.output_dir / "pdfs"
        self.images_dir = self.output_dir / "images" 
        self.texts_dir = self.output_dir / "texts"
        
        for dir_path in [self.pdfs_dir, self.images_dir, self.texts_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'total_papers': 0,
            'downloaded': 0,
            'processed': 0,
            'failed_downloads': 0,
            'failed_ocr': 0,
            'start_time': None,
            'processing_log': []
        }
        
        # Rate limiting
        self.download_delay = 16  # Seconds between downloads (respecting arXiv limits)
        self.last_download_time = 0
        
        # Session for downloads
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def load_paper_data(self, json_file: str = "arxiv_clean.json") -> List[Dict]:
        """Load paper data from arxiv_clean.json."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} papers from {json_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            return []
    
    def arxiv_url_to_pdf_url(self, arxiv_url: str) -> str:
        """Convert arXiv abstract URL to PDF download URL."""
        # Handle different arXiv URL formats
        if 'arxiv.org/abs/' in arxiv_url:
            paper_id = arxiv_url.split('/abs/')[-1]
        elif 'arxiv.org/pdf/' in arxiv_url:
            return arxiv_url
        else:
            # Extract paper ID using regex
            match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', arxiv_url)
            if match:
                paper_id = match.group(1)
            else:
                logger.warning(f"Could not extract paper ID from: {arxiv_url}")
                return arxiv_url
        
        # Remove version if present for URL construction
        base_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
        return f"https://arxiv.org/pdf/{base_id}.pdf"
    
    def download_pdf(self, pdf_url: str, paper_id: str) -> Optional[Path]:
        """Download PDF with rate limiting and error handling."""
        pdf_path = self.pdfs_dir / f"{paper_id}.pdf"
        
        # Skip if already downloaded
        if pdf_path.exists():
            logger.debug(f"PDF already exists: {pdf_path}")
            return pdf_path
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_download_time
        if time_since_last < self.download_delay:
            sleep_time = self.download_delay - time_since_last
            logger.info(f"Rate limiting: sleeping {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        try:
            logger.info(f"Downloading {pdf_url}")
            response = self.session.get(pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                logger.warning(f"Unexpected content type: {content_type} for {pdf_url}")
            
            # Save PDF
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.last_download_time = time.time()
            logger.info(f"Downloaded: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error downloading {pdf_url}: {e}")
            # Clean up partial file
            if pdf_path.exists():
                pdf_path.unlink()
            return None
    
    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements to improve OCR accuracy."""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            return image
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def pdf_to_images(self, pdf_path: Path, paper_id: str) -> List[Path]:
        """Convert PDF to images for OCR processing."""
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            
            # Convert PDF to images with high DPI for better OCR
            images = convert_from_path(
                pdf_path,
                dpi=300,  # High DPI for better text recognition
                first_page=None,
                last_page=None,
                fmt='PNG',
                thread_count=1,  # Conservative to avoid memory issues
                grayscale=True   # Grayscale for faster processing
            )
            
            image_paths = []
            
            for i, image in enumerate(images):
                # Enhance image for OCR
                enhanced_image = self.enhance_image_for_ocr(image)
                
                # Save image
                image_path = self.images_dir / f"{paper_id}_page_{i+1:03d}.png"
                enhanced_image.save(image_path, "PNG", optimize=True)
                image_paths.append(image_path)
                
                logger.debug(f"Saved page {i+1}: {image_path}")
            
            logger.info(f"Converted {len(images)} pages for {paper_id}")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def extract_text_with_layout(self, image_path: Path) -> str:
        """Extract text from image using Tesseract with layout preservation."""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Tesseract configuration for layout preservation
            # PSM 1: Automatic page segmentation with OSD
            # PSM 6: Assume uniform block of text
            # PSM 11: Sparse text (good for finding text in specific regions)
            custom_config = r'--oem 3 --psm 1 -c preserve_interword_spaces=1'
            
            # Extract text with layout information
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Alternative: Get detailed layout information
            # data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
            # This can be used for more sophisticated layout analysis
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR error for {image_path}: {e}")
            return ""
    
    def post_process_text(self, text: str, paper_title: str = "") -> str:
        """Post-process OCR text to improve structure and readability."""
        if not text:
            return text
        
        # Clean up common OCR artifacts
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines
        text = re.sub(r'([a-z])\n([A-Z])', r'\1 \2', text)  # Fix broken words
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        
        # Try to identify and format sections
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                processed_lines.append('')
                continue
            
            # Detect potential headers/titles (short lines, mostly uppercase)
            if len(line) < 100 and line.count(' ') < 8:
                if line.isupper() or (line[0].isupper() and sum(1 for c in line if c.isupper()) > len(line) * 0.3):
                    processed_lines.append(f"\n{line}\n")  # Add spacing around headers
                    continue
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def process_paper(self, paper_data: Dict) -> bool:
        """Process a single paper through the complete OCR pipeline."""
        paper_id = self.extract_paper_id(paper_data['url'])
        paper_title = paper_data.get('title', '')
        
        logger.info(f"Processing paper: {paper_id} - {paper_title[:50]}...")
        
        # Check if already processed
        output_file = self.texts_dir / f"{paper_id}.txt"
        if output_file.exists():
            logger.info(f"Already processed: {paper_id}")
            self.stats['processed'] += 1
            return True
        
        try:
            # Step 1: Download PDF
            pdf_url = self.arxiv_url_to_pdf_url(paper_data['url'])
            pdf_path = self.download_pdf(pdf_url, paper_id)
            
            if not pdf_path:
                self.stats['failed_downloads'] += 1
                self.log_processing_result(paper_id, paper_title, 'download_failed', '')
                return False
            
            self.stats['downloaded'] += 1
            
            # Step 2: Convert to images
            image_paths = self.pdf_to_images(pdf_path, paper_id)
            
            if not image_paths:
                self.stats['failed_ocr'] += 1
                self.log_processing_result(paper_id, paper_title, 'pdf_conversion_failed', '')
                return False
            
            # Step 3: OCR each page
            all_text = []
            
            for i, image_path in enumerate(image_paths):
                logger.debug(f"OCR processing page {i+1}/{len(image_paths)} for {paper_id}")
                page_text = self.extract_text_with_layout(image_path)
                
                if page_text:
                    all_text.append(f"=== PAGE {i+1} ===\n{page_text}\n")
                else:
                    all_text.append(f"=== PAGE {i+1} ===\n[OCR failed for this page]\n")
            
            # Step 4: Combine and post-process text
            combined_text = '\n'.join(all_text)
            processed_text = self.post_process_text(combined_text, paper_title)
            
            # Step 5: Save result
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {paper_title}\n")
                f.write(f"arXiv URL: {paper_data['url']}\n")
                f.write(f"Authors: {', '.join(paper_data.get('authors', []))}\n")
                f.write(f"Date: {paper_data.get('date', '')}\n")
                f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(processed_text)
            
            # Clean up temporary images to save space
            for image_path in image_paths:
                try:
                    image_path.unlink()
                except:
                    pass
            
            self.stats['processed'] += 1
            self.log_processing_result(paper_id, paper_title, 'success', str(output_file))
            
            logger.info(f"Successfully processed: {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {paper_id}: {e}")
            self.stats['failed_ocr'] += 1
            self.log_processing_result(paper_id, paper_title, 'processing_error', str(e))
            return False
    
    def extract_paper_id(self, url: str) -> str:
        """Extract clean paper ID from arXiv URL."""
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        if match:
            return match.group(1)
        else:
            # Fallback: use last part of URL
            return url.split('/')[-1].replace('v1', '').replace('.pdf', '')
    
    def log_processing_result(self, paper_id: str, title: str, status: str, output_path: str):
        """Log processing result for tracking."""
        self.stats['processing_log'].append({
            'paper_id': paper_id,
            'title': title[:100],
            'status': status,
            'output_path': output_path,
            'timestamp': datetime.now().isoformat()
        })
    
    def save_processing_log(self):
        """Save processing statistics and log."""
        log_file = self.output_dir / "processing_log.json"
        
        # Calculate timing
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            self.stats['total_time_seconds'] = elapsed
            self.stats['average_time_per_paper'] = elapsed / max(self.stats['processed'], 1)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing log saved to: {log_file}")
    
    def print_summary(self):
        """Print processing summary."""
        print(f"\n{'='*60}")
        print(f"BATCH OCR PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total papers: {self.stats['total_papers']}")
        print(f"Successfully downloaded: {self.stats['downloaded']}")
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"Failed downloads: {self.stats['failed_downloads']}")
        print(f"Failed OCR: {self.stats['failed_ocr']}")
        
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            print(f"Total time: {elapsed/60:.1f} minutes")
            if self.stats['processed'] > 0:
                print(f"Average time per paper: {elapsed/self.stats['processed']:.1f} seconds")
        
        print(f"Output directory: {self.output_dir}")
        print(f"Text files saved in: {self.texts_dir}")
        print(f"Processing log: {self.output_dir}/processing_log.json")
        print(f"{'='*60}")
    
    def run_batch_processing(self, json_file: str = "arxiv_clean.json", max_papers: Optional[int] = None):
        """Run the complete batch OCR processing pipeline."""
        logger.info("Starting batch OCR processing...")
        self.stats['start_time'] = time.time()
        
        # Load paper data
        papers = self.load_paper_data(json_file)
        if not papers:
            logger.error("No papers loaded. Exiting.")
            return
        
        if max_papers:
            papers = papers[:max_papers]
            logger.info(f"Limited to first {max_papers} papers")
        
        self.stats['total_papers'] = len(papers)
        
        # Process each paper
        for i, paper in enumerate(papers, 1):
            logger.info(f"\n--- Processing {i}/{len(papers)} ---")
            self.process_paper(paper)
            
            # Save progress periodically
            if i % 10 == 0:
                self.save_processing_log()
        
        # Final summary
        self.save_processing_log()
        self.print_summary()
        
        logger.info("Batch OCR processing completed!")

def main():
    """Main entry point."""
    processor = ArxivPDFOCR()
    
    # For testing, limit to first 5 papers
    # processor.run_batch_processing(max_papers=5)
    
    # For full processing
    processor.run_batch_processing()

if __name__ == "__main__":
    main()