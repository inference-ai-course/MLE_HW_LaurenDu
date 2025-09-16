import json
import random
from typing import List, Dict

def load_papers(json_file: str) -> List[Dict]:
    """Load papers from JSON file"""
    with open(json_file, 'r') as f:
        papers = json.load(f)
    return papers

def extract_key_terms(abstract: str) -> List[str]:
    """Extract key technical terms and concepts from abstract"""
    # Simple keyword extraction based on common academic patterns
    key_patterns = [
        'method', 'approach', 'algorithm', 'framework', 'model', 'system',
        'technique', 'pipeline', 'architecture', 'dataset', 'benchmark',
        'performance', 'accuracy', 'improvement', 'optimization', 'evaluation'
    ]
    
    terms = []
    words = abstract.lower().split()
    
    # Look for capitalized terms (likely important concepts)
    for word in abstract.split():
        if word[0].isupper() and len(word) > 3 and word.isalpha():
            terms.append(word)
    
    return terms[:5]  # Return top 5 key terms

def generate_generic_qa_for_paper(paper: Dict) -> List[Dict]:
    """Generate exactly 5 Q&A pairs for any paper based on its abstract"""
    title = paper['title']
    abstract = paper['abstract']
    categories = paper['categories']
    authors = paper['authors']
    
    qa_pairs = []
    
    # Extract key information from the abstract
    sentences = abstract.split('. ')
    first_sentence = sentences[0] if sentences else abstract[:200]
    
    # Always generate exactly 5 Q&A pairs
    
    # 1. Main contribution question (ALWAYS)
    qa_pairs.append({
        "question": f"What is the main contribution of the paper titled '{title}'?",
        "answer": f"The main contribution is {first_sentence.lower()}. {sentences[1] if len(sentences) > 1 else ''}"
    })
    
    # 2. Problem addressed question (ALWAYS)
    problem_sentence = None
    for sentence in sentences:
        if any(word in sentence.lower() for word in ['problem', 'challenge', 'limitation', 'addresses', 'gap', 'issue']):
            problem_sentence = sentence
            break
    
    if not problem_sentence:
        problem_sentence = f"This research addresses challenges in {categories.split(',')[0].strip()} by {first_sentence.lower()}"
    
    qa_pairs.append({
        "question": f"What problem or challenge does this research address?",
        "answer": problem_sentence.strip()
    })
    
    # 3. Method/approach question (ALWAYS)
    method_sentence = None
    for sentence in sentences:
        if any(word in sentence.lower() for word in ['method', 'approach', 'propose', 'introduce', 'framework', 'algorithm', 'technique']):
            method_sentence = sentence
            break
    
    if not method_sentence:
        method_sentence = f"The paper introduces {title.lower()}, which {sentences[1] if len(sentences) > 1 else 'addresses the research problem through novel techniques'}"
    
    qa_pairs.append({
        "question": f"What method or approach is proposed in this paper?",
        "answer": method_sentence.strip()
    })
    
    # 4. Performance/results question (ALWAYS)
    result_sentence = None
    for sentence in sentences:
        if any(word in sentence.lower() for word in ['achieve', 'performance', 'accuracy', 'improvement', 'results', 'outperform', 'demonstrate', 'show']):
            result_sentence = sentence
            break
    
    if not result_sentence:
        result_sentence = f"The paper demonstrates the effectiveness of their approach through {sentences[-1] if len(sentences) > 1 else 'experimental validation and analysis'}"
    
    qa_pairs.append({
        "question": f"What are the key results or performance achievements reported in this paper?",
        "answer": result_sentence.strip()
    })
    
    # 5. Application/domain question (ALWAYS)
    domain_mapping = {
        'cs.CV': 'computer vision',
        'cs.CL': 'natural language processing', 
        'cs.LG': 'machine learning',
        'cs.RO': 'robotics',
        'cs.AI': 'artificial intelligence',
        'cs.CR': 'cryptography and security',
        'cs.SE': 'software engineering',
        'cs.IR': 'information retrieval',
        'cs.HC': 'human-computer interaction',
        'cs.SD': 'sound processing',
        'math.': 'mathematics',
        'stat.': 'statistics',
        'eess.': 'electrical engineering'
    }
    
    domain = 'computer science'
    for key, value in domain_mapping.items():
        if key in categories:
            domain = value
            break
    
    qa_pairs.append({
        "question": f"What application domain or field does this research target?",
        "answer": f"This research targets {domain}, as indicated by its focus on {abstract.split('.')[0].lower()}."
    })
    
    # Add paper metadata to each Q&A pair
    for qa in qa_pairs:
        qa['paper_title'] = title
        qa['paper_categories'] = categories
        qa['type'] = 'regular'
    
    return qa_pairs

def generate_edge_case_qa(papers: List[Dict]) -> List[Dict]:
    """Generate edge case Q&A pairs that test model's ability to handle incorrect queries"""
    edge_cases = []
    
    # Sample papers for edge cases
    sample_papers = random.sample(papers, min(20, len(papers)))
    
    for i, paper in enumerate(sample_papers):
        title = paper['title']
        abstract = paper['abstract']
        
        # Generate different types of edge cases
        edge_case_types = [
            {
                "question": f"According to the paper '{title}', what is the exact computational complexity of their algorithm?",
                "answer": "The paper's abstract does not provide specific computational complexity analysis. The abstract focuses on the main contributions and results but does not include detailed algorithmic complexity information."
            },
            {
                "question": f"Does the paper '{title}' compare their method against BERT?",
                "answer": "The abstract does not mention comparisons with BERT. The paper may focus on different baselines or evaluation methods not detailed in the abstract provided."
            },
            {
                "question": f"What dataset from ImageNet does the paper '{title}' use for evaluation?",
                "answer": "The abstract does not specify the use of ImageNet datasets. The evaluation methodology and datasets used are not detailed in the abstract provided."
            }
        ]
        
        # Add one edge case per few papers
        if i % 3 == 0 and i < len(edge_case_types):
            edge_case = edge_case_types[i % len(edge_case_types)].copy()
            edge_case['paper_title'] = title
            edge_case['paper_categories'] = paper['categories']
            edge_case['type'] = 'edge_case'
            edge_cases.append(edge_case)
    
    return edge_cases

def convert_to_jsonl(qa_pairs: List[Dict], output_file: str):
    """Convert Q&A pairs to JSONL format for instruction tuning"""
    system_prompt = "You are a helpful academic Q&A assistant specialized in scholarly content."
    
    with open(output_file, 'w') as f:
        for qa in qa_pairs:
            user_q = qa["question"]
            assistant_a = qa["answer"]
            
            # Compose the prompt with system, user, assistant roles
            full_prompt = f"<|system|>{system_prompt}<|user|>{user_q}<|assistant|>{assistant_a}"
            
            entry = {"text": full_prompt}
            f.write(json.dumps(entry) + "\n")

def main():
    # Load papers
    papers = load_papers("parsed_papers.json")
    print(f"Loaded {len(papers)} papers")
    
    # Generate Q&A pairs for first 100 papers to create ~500 Q&A pairs
    selected_papers = papers[:100]
    
    all_qa_pairs = []
    
    print("Generating Q&A pairs...")
    for i, paper in enumerate(selected_papers):
        print(f"Processing paper {i+1}/100: {paper['title'][:60]}...")
        qa_pairs = generate_generic_qa_for_paper(paper)
        all_qa_pairs.extend(qa_pairs)
    
    # Generate edge cases
    print("Generating edge case Q&A pairs...")
    edge_cases = generate_edge_case_qa(selected_papers)
    all_qa_pairs.extend(edge_cases)
    
    # Save raw Q&A pairs
    with open("qa_pairs_comprehensive.json", 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"Generated {len(all_qa_pairs)} Q&A pairs")
    
    # Convert to JSONL format
    convert_to_jsonl(all_qa_pairs, "synthetic_qa_comprehensive.jsonl")
    
    print("Q&A generation complete!")
    print(f"Raw Q&A pairs saved to: qa_pairs_comprehensive.json")
    print(f"JSONL training data saved to: synthetic_qa_comprehensive.jsonl")
    
    # Print statistics
    regular_count = sum(1 for qa in all_qa_pairs if qa.get('type') == 'regular')
    edge_count = sum(1 for qa in all_qa_pairs if qa.get('type') == 'edge_case')
    print(f"Regular Q&A pairs: {regular_count}")
    print(f"Edge case Q&A pairs: {edge_count}")

if __name__ == "__main__":
    main()