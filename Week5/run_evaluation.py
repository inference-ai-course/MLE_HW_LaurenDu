#!/usr/bin/env python3
"""
Script to run comprehensive evaluation of hybrid search methods
"""

import sys
import logging
from pathlib import Path
from evaluation import SearchEvaluator
from rag_pipeline import RAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the evaluation pipeline"""
    try:
        logger.info("="*60)
        logger.info("HYBRID SEARCH EVALUATION")
        logger.info("="*60)
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        pipeline.run_pipeline()
        
        logger.info(f"Pipeline loaded with {len(pipeline.chunks)} chunks from {len(set(m['pdf_file'] for m in pipeline.chunk_metadata))} papers")
        
        # Create evaluator
        logger.info("Creating evaluator...")
        evaluator = SearchEvaluator(pipeline)
        
        # Run evaluation
        logger.info("Running comprehensive evaluation...")
        logger.info("This may take a few minutes...")
        
        results = evaluator.run_full_evaluation()
        
        # Generate reports
        logger.info("Generating comparison report...")
        df = evaluator.create_comparison_report(results, "detailed_evaluation_results.json")
        
        # Save summary
        df.to_csv("evaluation_summary.csv", index=False)
        
        # Display results
        print("\n" + "="*80)
        print("HYBRID SEARCH EVALUATION RESULTS")
        print("="*80)
        print(df.round(3).to_string(index=False))
        
        # Create visualizations
        logger.info("Creating visualizations...")
        evaluator.plot_comparison(df, "evaluation_comparison.png")
        
        # Generate detailed analysis
        logger.info("Generating detailed analysis...")
        analysis = evaluator.generate_detailed_analysis(results)
        with open("evaluation_analysis.md", "w") as f:
            f.write(analysis)
        
        # Summary of best methods
        print("\n" + "="*80)
        print("BEST PERFORMING METHODS BY METRIC")
        print("="*80)
        
        key_metrics = ['hit_rate@3', 'precision@3', 'recall@3', 'hit_rate@5']
        
        for metric in key_metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_method = df.loc[best_idx, 'method']
                best_score = df.loc[best_idx, metric]
                print(f"📊 {metric:<15}: {best_method:<25} ({best_score:.3f})")
        
        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        # Find overall best hybrid method
        hybrid_methods = df[df['method'].str.contains('hybrid')]
        if not hybrid_methods.empty:
            overall_best_idx = hybrid_methods['hit_rate@3'].idxmax()
            best_hybrid = hybrid_methods.loc[overall_best_idx, 'method']
            best_score = hybrid_methods.loc[overall_best_idx, 'hit_rate@3']
            
            print(f"🎯 Best hybrid method: {best_hybrid} (Hit Rate@3: {best_score:.3f})")
        
        # Compare to baselines
        vector_score = df[df['method'] == 'vector']['hit_rate@3'].iloc[0] if not df[df['method'] == 'vector'].empty else 0
        fts_score = df[df['method'] == 'keyword_fts']['hit_rate@3'].iloc[0] if not df[df['method'] == 'keyword_fts'].empty else 0
        
        if not hybrid_methods.empty:
            best_hybrid_score = hybrid_methods['hit_rate@3'].max()
            
            if best_hybrid_score > max(vector_score, fts_score):
                improvement = best_hybrid_score - max(vector_score, fts_score)
                print(f"✅ Hybrid search improves over best baseline by {improvement:.3f} ({improvement/max(vector_score, fts_score)*100:.1f}%)")
            else:
                print("⚠️  Hybrid search does not consistently outperform baselines")
        
        # File outputs
        print("\n" + "="*80)
        print("OUTPUT FILES CREATED")
        print("="*80)
        print("📄 detailed_evaluation_results.json - Complete evaluation results")
        print("📊 evaluation_summary.csv - Summary metrics table")
        print("📈 evaluation_comparison.png - Performance comparison chart")
        print("📝 evaluation_analysis.md - Detailed analysis report")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()