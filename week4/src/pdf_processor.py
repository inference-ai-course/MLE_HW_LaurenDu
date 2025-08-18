import fitz
import json
import os
import re
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_directory: str, output_directory: str):
        self.pdf_directory = pdf_directory
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\']+', '', text)
        text = text.strip()
        return text
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                full_text += text + "\n"
            
            doc.close()
            return self.clean_text(full_text)
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return ""
    
    def process_all_pdfs(self) -> Dict[str, str]:
        documents = {}
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        pdf_files.sort(key=lambda x: int(x.split('.')[0]))
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            logger.info(f"Processing {pdf_file}...")
            
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                documents[pdf_file] = text
                logger.info(f"Successfully processed {pdf_file} ({len(text)} characters)")
            else:
                logger.warning(f"No text extracted from {pdf_file}")
        
        output_path = os.path.join(self.output_directory, "processed_documents.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(documents)} processed documents to {output_path}")
        return documents

if __name__ == "__main__":
    processor = PDFProcessor(
        pdf_directory="../PDFs",
        output_directory="../data"
    )
    documents = processor.process_all_pdfs()
    print(f"Processed {len(documents)} documents successfully!")