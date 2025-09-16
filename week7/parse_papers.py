#!/usr/bin/env python3
"""
Parse arXiv papers from the selected_arxiv_papers_100.txt file and extract structured data.
"""

import re
import json
from typing import List, Dict

def parse_arxiv_papers(file_path: str) -> List[Dict]:
    """
    Parse the arXiv papers file and extract structured data for each paper.
    
    Args:
        file_path: Path to the selected_arxiv_papers_100.txt file
        
    Returns:
        List of dictionaries containing paper data
    """
    papers = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by paper markers
    paper_sections = re.split(r'PAPER #\d+', content)[1:]  # Skip the header
    
    for section in paper_sections:
        paper_data = {}
        lines = section.strip().split('\n')
        
        # Extract title
        title_match = re.search(r'Title: (.+)', section)
        if title_match:
            paper_data['title'] = title_match.group(1).strip()
        
        # Extract link
        link_match = re.search(r'Link: (.+)', section)
        if link_match:
            paper_data['link'] = link_match.group(1).strip()
        
        # Extract authors
        authors_match = re.search(r'Authors: (.+)', section)
        if authors_match:
            paper_data['authors'] = authors_match.group(1).strip()
        
        # Extract published date
        published_match = re.search(r'Published: (.+)', section)
        if published_match:
            paper_data['published'] = published_match.group(1).strip()
        
        # Extract categories
        categories_match = re.search(r'Categories: (.+)', section)
        if categories_match:
            paper_data['categories'] = categories_match.group(1).strip()
        
        # Extract abstract (everything after "Abstract:" until next section or end)
        abstract_match = re.search(r'Abstract:\s*\n(.*?)(?=\n=+|$)', section, re.DOTALL)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            # Clean up the abstract text
            abstract_text = re.sub(r'\n+', ' ', abstract_text)
            abstract_text = re.sub(r'\s+', ' ', abstract_text)
            paper_data['abstract'] = abstract_text
        
        # Only add papers that have both title and abstract
        if 'title' in paper_data and 'abstract' in paper_data:
            papers.append(paper_data)
    
    return papers

def save_parsed_papers(papers: List[Dict], output_path: str):
    """Save parsed papers to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

def main():
    input_file = 'selected_arxiv_papers_100.txt'
    output_file = 'parsed_papers.json'
    
    print(f"Parsing papers from {input_file}...")
    papers = parse_arxiv_papers(input_file)
    
    print(f"Successfully parsed {len(papers)} papers")
    
    # Save to JSON file
    save_parsed_papers(papers, output_file)
    print(f"Saved parsed papers to {output_file}")
    
    # Display sample paper for verification
    if papers:
        print("\nSample paper:")
        print(f"Title: {papers[0]['title']}")
        print(f"Categories: {papers[0]['categories']}")
        print(f"Abstract: {papers[0]['abstract'][:200]}...")

if __name__ == "__main__":
    main()