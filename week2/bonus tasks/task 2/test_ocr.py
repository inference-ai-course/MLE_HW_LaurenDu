#!/usr/bin/env python3
"""
Quick test of OCR processing pipeline
"""

import json
from pdf_ocr_processor import ArxivPDFOCR

def quick_test():
    """Test OCR processing with just 2 papers."""
    processor = ArxivPDFOCR()
    
    # Load papers
    papers = processor.load_paper_data("arxiv_clean.json")
    
    if not papers:
        print("❌ No papers found in arxiv_clean.json")
        return
    
    print(f"✅ Loaded {len(papers)} papers")
    print(f"🧪 Testing with first 2 papers only...")
    
    # Test with just 2 papers
    processor.run_batch_processing(max_papers=2)

if __name__ == "__main__":
    quick_test()