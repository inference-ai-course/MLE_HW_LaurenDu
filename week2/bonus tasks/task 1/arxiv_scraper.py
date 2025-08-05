#!/usr/bin/env python3
"""
arXiv Paper Abstract Scraper
Fetches latest 200 papers from cs.CL category and extracts abstracts using multiple methods.
"""

import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import re
import os

# Core dependencies
import requests
from bs4 import BeautifulSoup
import trafilatura

# arXiv API
import arxiv

# OCR and screenshots
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("Warning: OCR dependencies not available. Install selenium, pytesseract, pillow for full functionality.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivScraper:
    """arXiv Paper Abstract Scraper with multiple extraction methods."""
    
    def __init__(self):
        self.papers = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Setup OCR if available
        self.ocr_driver = None
        if HAS_OCR:
            self._setup_ocr_driver()
    
    def _setup_ocr_driver(self):
        """Initialize Selenium driver for screenshots."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--window-size=1920,1080')
            
            service = Service(ChromeDriverManager().install())
            self.ocr_driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("OCR driver initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize OCR driver: {e}")
            self.ocr_driver = None
    
    def fetch_arxiv_papers(self, max_results: int = 200) -> List[Dict]:
        """Fetch papers from arXiv API."""
        logger.info(f"Fetching {max_results} papers from cs.CL category...")
        
        search = arxiv.Search(
            query="cat:cs.CL",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        client = arxiv.Client()
        papers = []
        
        try:
            for paper in client.results(search):
                paper_data = {
                    'url': paper.entry_id,
                    'title': paper.title.strip(),
                    'authors': [author.name for author in paper.authors],
                    'date': paper.published.strftime('%Y-%m-%d'),
                    'abstract': paper.summary.strip() if paper.summary else '',
                    'source': 'arxiv_api'
                }
                papers.append(paper_data)
                
        except Exception as e:
            logger.error(f"Error fetching from arXiv API: {e}")
            
        logger.info(f"Fetched {len(papers)} papers from arXiv API")
        return papers
    
    def scrape_abstract_page(self, paper_url: str) -> Optional[str]:
        """Scrape abstract from arXiv /abs/ page using requests + BeautifulSoup."""
        try:
            # Convert to /abs/ URL if needed
            abs_url = paper_url.replace('/pdf/', '/abs/').replace('.pdf', '')
            if not abs_url.startswith('https://arxiv.org/abs/'):
                # Extract paper ID and construct proper URL
                paper_id = abs_url.split('/')[-1]
                abs_url = f"https://arxiv.org/abs/{paper_id}"
            
            logger.debug(f"Scraping: {abs_url}")
            
            # Add delay to be respectful
            time.sleep(1)
            
            response = self.session.get(abs_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find abstract section
            abstract_div = soup.find('blockquote', class_='abstract')
            if abstract_div:
                abstract_text = abstract_div.get_text(strip=True)
                # Remove "Abstract:" prefix if present
                abstract_text = re.sub(r'^Abstract:\s*', '', abstract_text)
                return abstract_text
                
        except Exception as e:
            logger.debug(f"Error scraping {paper_url}: {e}")
            
        return None
    
    def extract_with_trafilatura(self, paper_url: str) -> Optional[str]:
        """Extract content using Trafilatura."""
        try:
            abs_url = paper_url.replace('/pdf/', '/abs/').replace('.pdf', '')
            if not abs_url.startswith('https://arxiv.org/abs/'):
                paper_id = abs_url.split('/')[-1]
                abs_url = f"https://arxiv.org/abs/{paper_id}"
            
            time.sleep(1)  # Rate limiting
            
            downloaded = trafilatura.fetch_url(abs_url)
            if downloaded:
                extracted = trafilatura.extract(downloaded)
                if extracted and len(extracted) > 50:  # Minimum length check
                    # Try to extract just the abstract portion
                    lines = extracted.split('\n')
                    for i, line in enumerate(lines):
                        if 'abstract' in line.lower():
                            # Take next few lines as abstract
                            abstract_lines = lines[i+1:i+10]
                            abstract = ' '.join(abstract_lines).strip()
                            if len(abstract) > 50:
                                return abstract[:2000]  # Limit length
                    
                    # Fallback: return first substantial paragraph
                    paragraphs = [p.strip() for p in extracted.split('\n\n') if len(p.strip()) > 50]
                    if paragraphs:
                        return paragraphs[0][:2000]
                        
        except Exception as e:
            logger.debug(f"Trafilatura extraction error for {paper_url}: {e}")
            
        return None
    
    def ocr_extract_abstract(self, paper_url: str) -> Optional[str]:
        """Extract abstract using OCR from screenshot."""
        if not self.ocr_driver:
            return None
            
        try:
            abs_url = paper_url.replace('/pdf/', '/abs/').replace('.pdf', '')
            if not abs_url.startswith('https://arxiv.org/abs/'):
                paper_id = abs_url.split('/')[-1]
                abs_url = f"https://arxiv.org/abs/{paper_id}"
            
            logger.debug(f"OCR extraction for: {abs_url}")
            
            self.ocr_driver.get(abs_url)
            time.sleep(3)  # Wait for page load
            
            # Take screenshot
            screenshot_path = f"/tmp/arxiv_screenshot_{int(time.time())}.png"
            self.ocr_driver.save_screenshot(screenshot_path)
            
            # Perform OCR
            image = Image.open(screenshot_path)
            text = pytesseract.image_to_string(image, config='--psm 6')
            
            # Clean up
            os.remove(screenshot_path)
            
            # Extract abstract from OCR text
            if 'abstract' in text.lower():
                lines = text.split('\n')
                abstract_started = False
                abstract_lines = []
                
                for line in lines:
                    line = line.strip()
                    if 'abstract' in line.lower() and not abstract_started:
                        abstract_started = True
                        # Include this line if it has more than just "Abstract"
                        if len(line) > 10:
                            abstract_lines.append(line)
                        continue
                    
                    if abstract_started:
                        if line and not line.lower().startswith(('keywords', 'categories', 'comments')):
                            abstract_lines.append(line)
                        elif line.lower().startswith(('keywords', 'categories', 'comments')):
                            break
                
                if abstract_lines:
                    abstract = ' '.join(abstract_lines)
                    # Clean up OCR artifacts
                    abstract = re.sub(r'[^\w\s\.,;:!?\-()]', '', abstract)
                    return abstract[:2000] if len(abstract) > 50 else None
            
        except Exception as e:
            logger.debug(f"OCR extraction error for {paper_url}: {e}")
            
        return None
    
    def enhance_paper_abstracts(self, papers: List[Dict]) -> List[Dict]:
        """Enhance papers with better abstracts using scraping and OCR."""
        logger.info("Enhancing abstracts with additional extraction methods...")
        
        enhanced_papers = []
        
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper['title'][:50]}...")
            
            enhanced_paper = paper.copy()
            best_abstract = paper.get('abstract', '')
            
            # Method 1: Direct scraping
            scraped_abstract = self.scrape_abstract_page(paper['url'])
            if scraped_abstract and len(scraped_abstract) > len(best_abstract):
                best_abstract = scraped_abstract
                enhanced_paper['extraction_method'] = 'scraping'
            
            # Method 2: Trafilatura
            if len(best_abstract) < 100:  # Only if we don't have a good abstract yet
                trafilatura_abstract = self.extract_with_trafilatura(paper['url'])
                if trafilatura_abstract and len(trafilatura_abstract) > len(best_abstract):
                    best_abstract = trafilatura_abstract
                    enhanced_paper['extraction_method'] = 'trafilatura'
            
            # Method 3: OCR (fallback)
            if len(best_abstract) < 50 and self.ocr_driver:
                ocr_abstract = self.ocr_extract_abstract(paper['url'])
                if ocr_abstract and len(ocr_abstract) > len(best_abstract):
                    best_abstract = ocr_abstract
                    enhanced_paper['extraction_method'] = 'ocr'
            
            enhanced_paper['abstract'] = best_abstract
            
            # Remove source field for final output
            enhanced_paper.pop('source', None)
            enhanced_paper.pop('extraction_method', None)
            
            enhanced_papers.append(enhanced_paper)
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(papers)} papers")
        
        return enhanced_papers
    
    def optimize_output_size(self, papers: List[Dict], max_size_mb: float = 1.0) -> List[Dict]:
        """Optimize output to stay under size limit."""
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # Calculate current size
        current_data = json.dumps(papers, indent=2)
        current_size = len(current_data.encode('utf-8'))
        
        logger.info(f"Current data size: {current_size / 1024 / 1024:.2f} MB")
        
        if current_size <= max_size_bytes:
            return papers
        
        logger.info("Optimizing data size...")
        
        # Optimization strategies
        optimized_papers = []
        for paper in papers:
            optimized_paper = {
                'url': paper['url'],
                'title': paper['title'][:200],  # Limit title length
                'abstract': paper['abstract'][:1000],  # Limit abstract length
                'authors': paper['authors'][:5],  # Limit number of authors
                'date': paper['date']
            }
            optimized_papers.append(optimized_paper)
            
            # Check size after each paper
            temp_data = json.dumps(optimized_papers, indent=2)
            temp_size = len(temp_data.encode('utf-8'))
            
            if temp_size > max_size_bytes:
                # Remove last paper and break
                optimized_papers.pop()
                break
        
        final_size = len(json.dumps(optimized_papers, indent=2).encode('utf-8'))
        logger.info(f"Optimized to {len(optimized_papers)} papers, {final_size / 1024 / 1024:.2f} MB")
        
        return optimized_papers
    
    def save_results(self, papers: List[Dict], filename: str = "arxiv_clean.json"):
        """Save results to JSON file."""
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(filepath)
        logger.info(f"Saved {len(papers)} papers to {filepath}")
        logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
        
        return filepath
    
    def run(self):
        """Main execution method."""
        logger.info("Starting arXiv Abstract Scraper...")
        
        try:
            # Step 1: Fetch papers from arXiv API
            papers = self.fetch_arxiv_papers(max_results=200)
            
            if not papers:
                logger.error("No papers fetched from arXiv API")
                return
            
            # Step 2: Enhance abstracts with additional methods
            enhanced_papers = self.enhance_paper_abstracts(papers)
            
            # Step 3: Optimize for size limit  
            optimized_papers = self.optimize_output_size(enhanced_papers)
            
            # Step 4: Save results
            output_file = self.save_results(optimized_papers)
            
            logger.info(f"Scraping completed successfully!")
            logger.info(f"Output saved to: {output_file}")
            
            # Print summary
            print(f"\n=== SCRAPING SUMMARY ===")
            print(f"Papers processed: {len(optimized_papers)}")
            print(f"Output file: {output_file}")
            print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            raise
        
        finally:
            # Clean up
            if self.ocr_driver:
                self.ocr_driver.quit()

def main():
    """Main entry point."""
    scraper = ArxivScraper()
    scraper.run()

if __name__ == "__main__":
    main()
