import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from rag_pipeline import RAGPipeline
from hybrid_search import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchEvaluator:
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize evaluator with RAG pipeline
        
        Args:
            rag_pipeline: Initialized RAGPipeline instance
        """
        self.pipeline = rag_pipeline
        self.hybrid_retriever = HybridRetriever(rag_pipeline)
        self.test_queries = self._create_test_queries()
    
    def _create_test_queries(self) -> List[Dict]:
        """
        Create test queries with known relevant documents
        
        Returns:
            List of test query dictionaries
        """
        return [
            {
                "query": "neural networks machine learning",
                "relevant_papers": ["1.pdf", "2.pdf", "18.pdf", "19.pdf"],
                "description": "General machine learning and neural networks"
            },
            {
                "query": "natural language processing NLP",
                "relevant_papers": ["1.pdf", "3.pdf", "8.pdf", "27.pdf", "30.pdf"],
                "description": "Natural language processing techniques"
            },
            {
                "query": "parsing grammar syntactic analysis",
                "relevant_papers": ["3.pdf", "4.pdf", "5.pdf", "14.pdf", "24.pdf"],
                "description": "Syntactic parsing and grammar"
            },
            {
                "query": "speech recognition spoken language",
                "relevant_papers": ["18.pdf", "28.pdf", "31.pdf", "35.pdf", "49.pdf"],
                "description": "Speech processing and recognition"
            },
            {
                "query": "machine translation translation systems",
                "relevant_papers": ["15.pdf", "16.pdf", "29.pdf", "40.pdf", "44.pdf", "48.pdf"],
                "description": "Machine translation approaches"
            },
            {
                "query": "text segmentation discourse structure",
                "relevant_papers": ["1.pdf", "13.pdf", "21.pdf", "41.pdf"],
                "description": "Text segmentation and discourse"
            },
            {
                "query": "word sense disambiguation lexical semantics",
                "relevant_papers": ["20.pdf", "25.pdf", "26.pdf", "38.pdf"],
                "description": "Word sense and lexical semantics"
            },
            {
                "query": "information retrieval IR text retrieval",
                "relevant_papers": ["43.pdf", "50.pdf"],
                "description": "Information retrieval systems"
            },
            {
                "query": "dialogue systems conversational agents",
                "relevant_papers": ["2.pdf", "28.pdf", "30.pdf", "35.pdf", "39.pdf"],
                "description": "Dialog and conversation systems"
            },
            {
                "query": "part of speech tagging POS morphological analysis",
                "relevant_papers": ["8.pdf", "11.pdf", "12.pdf", "17.pdf"],
                "description": "POS tagging and morphology"
            },
            {
                "query": "probabilistic models statistical methods",
                "relevant_papers": ["4.pdf", "5.pdf", "20.pdf", "34.pdf"],
                "description": "Statistical and probabilistic approaches"
            },
            {
                "query": "corpus annotation linguistic annotation",
                "relevant_papers": ["27.pdf", "41.pdf", "47.pdf"],
                "description": "Corpus and linguistic annotation"
            }
        ]
    
    def evaluate_method(self, method_name: str, search_func, k_values: List[int] = [3, 5, 10]) -> Dict:
        """
        Evaluate a search method
        
        Args:
            method_name: Name of the search method
            search_func: Function that takes (query, k) and returns results
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'method': method_name,
            'queries': [],
            'metrics': {}
        }
        
        for query_info in self.test_queries:
            query = query_info['query']
            relevant_papers = set(query_info['relevant_papers'])
            
            query_results = {
                'query': query,
                'description': query_info['description'],
                'relevant_count': len(relevant_papers),
                'results_by_k': {}
            }
            
            for k in k_values:
                try:
                    search_results = search_func(query, k)
                    retrieved_papers = set()
                    
                    # Extract paper IDs from results
                    for result in search_results:
                        if 'metadata' in result:
                            retrieved_papers.add(result['metadata']['pdf_file'])
                        elif 'paper' in result:
                            retrieved_papers.add(result['paper'])
                    
                    # Calculate metrics
                    relevant_retrieved = relevant_papers & retrieved_papers
                    
                    precision = len(relevant_retrieved) / len(retrieved_papers) if retrieved_papers else 0
                    recall = len(relevant_retrieved) / len(relevant_papers) if relevant_papers else 0
                    hit_rate = 1.0 if len(relevant_retrieved) > 0 else 0.0
                    
                    query_results['results_by_k'][k] = {
                        'precision': precision,
                        'recall': recall,
                        'hit_rate': hit_rate,
                        'relevant_retrieved': len(relevant_retrieved),
                        'total_retrieved': len(retrieved_papers),
                        'retrieved_papers': list(retrieved_papers),
                        'relevant_retrieved_papers': list(relevant_retrieved)
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating {method_name} for query '{query}': {e}")
                    query_results['results_by_k'][k] = {
                        'precision': 0, 'recall': 0, 'hit_rate': 0,
                        'relevant_retrieved': 0, 'total_retrieved': 0,
                        'retrieved_papers': [], 'relevant_retrieved_papers': []
                    }
            
            results['queries'].append(query_results)
        
        # Calculate aggregate metrics
        for k in k_values:
            precisions = [q['results_by_k'][k]['precision'] for q in results['queries']]
            recalls = [q['results_by_k'][k]['recall'] for q in results['queries']]
            hit_rates = [q['results_by_k'][k]['hit_rate'] for q in results['queries']]
            
            results['metrics'][f'precision@{k}'] = sum(precisions) / len(precisions)
            results['metrics'][f'recall@{k}'] = sum(recalls) / len(recalls)
            results['metrics'][f'hit_rate@{k}'] = sum(hit_rates) / len(hit_rates)
        
        return results
    
    def run_full_evaluation(self) -> Dict[str, Dict]:
        """
        Run evaluation on all search methods
        
        Returns:
            Dictionary with results for each method
        """
        logger.info("Starting full evaluation...")
        
        # Define search methods
        methods = {
            'vector': lambda q, k: self.hybrid_retriever.vector_search(q, k),
            'keyword_fts': lambda q, k: self.hybrid_retriever.keyword_search_fts(q, k),
            'keyword_bm25': lambda q, k: self.hybrid_retriever.keyword_search_bm25(q, k),
            'hybrid_weighted_0.3': lambda q, k: self.hybrid_retriever.hybrid_search(q, k, alpha=0.3, merge_method='weighted'),
            'hybrid_weighted_0.5': lambda q, k: self.hybrid_retriever.hybrid_search(q, k, alpha=0.5, merge_method='weighted'),
            'hybrid_weighted_0.7': lambda q, k: self.hybrid_retriever.hybrid_search(q, k, alpha=0.7, merge_method='weighted'),
            'hybrid_rrf': lambda q, k: self.hybrid_retriever.hybrid_search(q, k, merge_method='rrf')
        }
        
        all_results = {}
        
        for method_name, search_func in methods.items():
            logger.info(f"Evaluating {method_name}...")
            all_results[method_name] = self.evaluate_method(method_name, search_func)
        
        return all_results
    
    def create_comparison_report(self, results: Dict[str, Dict], output_file: str = "evaluation_report.json") -> pd.DataFrame:
        """
        Create a comparison report of all methods
        
        Args:
            results: Results from run_full_evaluation
            output_file: File to save detailed results
            
        Returns:
            DataFrame with summary metrics
        """
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        
        for method_name, method_results in results.items():
            row = {'method': method_name}
            row.update(method_results['metrics'])
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Sort columns for better readability
        metric_cols = [col for col in df.columns if col != 'method']
        metric_cols.sort()
        df = df[['method'] + metric_cols]
        
        return df
    
    def plot_comparison(self, df: pd.DataFrame, save_path: str = "evaluation_comparison.png"):
        """
        Create visualizations comparing different methods
        
        Args:
            df: Summary DataFrame from create_comparison_report
            save_path: Path to save the plot
        """
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Search Method Comparison', fontsize=16, fontweight='bold')
        
        # Metrics to plot
        metrics_to_plot = ['precision@3', 'recall@3', 'hit_rate@3', 'hit_rate@5']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]
            
            if metric in df.columns:
                # Create bar plot
                bars = ax.bar(range(len(df)), df[metric], alpha=0.8)
                ax.set_xlabel('Method')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df['method'], rotation=45, ha='right')
                ax.set_ylim(0, 1.0)
                
                # Color bars based on performance
                for j, bar in enumerate(bars):
                    value = df.iloc[j][metric]
                    if value >= 0.7:
                        bar.set_color('green')
                    elif value >= 0.4:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
                
                # Add value labels on bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Comparison plot saved to {save_path}")
    
    def generate_detailed_analysis(self, results: Dict[str, Dict]) -> str:
        """
        Generate detailed textual analysis of results
        
        Args:
            results: Results from run_full_evaluation
            
        Returns:
            Detailed analysis as string
        """
        analysis = ["# Search Method Evaluation Report\n"]
        
        # Overall summary
        analysis.append("## Summary\n")
        analysis.append(f"Evaluated {len(results)} search methods on {len(self.test_queries)} test queries.\n")
        
        # Best performing methods
        best_methods = {}
        for metric in ['precision@3', 'recall@3', 'hit_rate@3']:
            best_method = max(results.items(), key=lambda x: x[1]['metrics'].get(metric, 0))
            best_methods[metric] = (best_method[0], best_method[1]['metrics'].get(metric, 0))
        
        analysis.append("### Best Performing Methods:\n")
        for metric, (method, score) in best_methods.items():
            analysis.append(f"- **{metric}**: {method} ({score:.3f})\n")
        
        # Method-by-method analysis
        analysis.append("\n## Method Analysis\n")
        
        for method_name, method_results in results.items():
            analysis.append(f"### {method_name}\n")
            metrics = method_results['metrics']
            analysis.append(f"- Precision@3: {metrics.get('precision@3', 0):.3f}\n")
            analysis.append(f"- Recall@3: {metrics.get('recall@3', 0):.3f}\n")
            analysis.append(f"- Hit Rate@3: {metrics.get('hit_rate@3', 0):.3f}\n")
            analysis.append(f"- Hit Rate@5: {metrics.get('hit_rate@5', 0):.3f}\n\n")
        
        # Query-specific insights
        analysis.append("## Query-Specific Insights\n")
        
        for i, query_info in enumerate(self.test_queries):
            query = query_info['query']
            analysis.append(f"### Query: \"{query}\"\n")
            
            # Find best method for this query
            best_hit_rate = 0
            best_method = None
            
            for method_name, method_results in results.items():
                hit_rate = method_results['queries'][i]['results_by_k'][3]['hit_rate']
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_method = method_name
            
            analysis.append(f"- Best method: {best_method} (Hit Rate: {best_hit_rate:.3f})\n")
            analysis.append(f"- Relevant papers: {len(query_info['relevant_papers'])}\n\n")
        
        return "".join(analysis)

def main():
    """Main evaluation function"""
    logger.info("Starting evaluation...")
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    pipeline.run_pipeline()
    
    # Create evaluator
    evaluator = SearchEvaluator(pipeline)
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Create comparison report
    df = evaluator.create_comparison_report(results, "detailed_evaluation_results.json")
    
    # Display summary
    print("\n" + "="*60)
    print("SEARCH METHOD EVALUATION SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save summary to CSV
    df.to_csv("evaluation_summary.csv", index=False)
    print(f"\nSummary saved to evaluation_summary.csv")
    
    # Create plots
    evaluator.plot_comparison(df)
    
    # Generate detailed analysis
    analysis = evaluator.generate_detailed_analysis(results)
    with open("evaluation_analysis.md", "w") as f:
        f.write(analysis)
    print("Detailed analysis saved to evaluation_analysis.md")
    
    # Print top performing methods
    print("\n" + "="*60)
    print("TOP PERFORMING METHODS")
    print("="*60)
    
    for metric in ['hit_rate@3', 'precision@3', 'recall@3']:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_method = df.loc[best_idx, 'method']
            best_score = df.loc[best_idx, metric]
            print(f"{metric}: {best_method} ({best_score:.3f})")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()