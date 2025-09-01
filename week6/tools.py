"""
Tool functions for the voice assistant to perform specific tasks.
These functions are called when the LLM detects the need for external tool usage.
"""

import json
import requests
from typing import Dict, Any


def search_arxiv(query: str) -> str:
    """
    Simulate an arXiv search or return a dummy passage for the given query.
    In a real system, this might query the arXiv API and extract a summary.
    
    Args:
        query (str): The search query for arXiv papers
        
    Returns:
        str: A formatted response with search results or mock data
    """
    try:
        # Mock arXiv search results for demonstration
        # In a production system, you would integrate with the actual arXiv API
        mock_results = {
            "quantum entanglement": "Recent research on quantum entanglement shows promising applications in quantum computing and cryptography. Key findings include improved entanglement generation methods and novel applications in quantum teleportation.",
            "machine learning": "Current machine learning research focuses on transformer architectures, few-shot learning, and ethical AI. Notable developments include improved language models and advances in computer vision.",
            "neural networks": "Neural network research continues to evolve with attention mechanisms, regularization techniques, and novel architectures. Recent papers explore efficient training methods and interpretability.",
            "deep learning": "Deep learning advances include new optimization algorithms, architectural innovations, and applications in various domains. Research shows progress in both theoretical understanding and practical applications.",
            "artificial intelligence": "AI research encompasses machine learning, natural language processing, computer vision, and robotics. Recent work focuses on responsible AI development and cross-domain applications."
        }
        
        query_lower = query.lower()
        
        # Check for keyword matches in mock database
        for keyword, result in mock_results.items():
            if keyword in query_lower:
                return result
        
        # Default response for unmatched queries
        return f"Found several relevant papers about {query}. The research shows ongoing developments with practical applications and theoretical contributions."
        
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.
    Uses sympy for safe mathematical evaluation.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        str: The result of the calculation or an error message
    """
    try:
        from sympy import sympify, latex, simplify
        from sympy.parsing.sympy_parser import parse_expr
        import re
        
        # Clean up the expression
        expression = expression.strip()
        
        # Handle common mathematical expressions and convert to sympy format
        # Replace common patterns
        expression = re.sub(r'(\d+)\s*\*\*\s*(\d+)', r'\1**\2', expression)  # Fix exponents
        expression = re.sub(r'(\w+)\s*squared', r'\1**2', expression)  # "x squared" -> "x**2"
        expression = re.sub(r'(\w+)\s*cubed', r'\1**3', expression)    # "x cubed" -> "x**3"
        expression = re.sub(r'square\s+root\s+of\s+(\w+)', r'sqrt(\1)', expression)  # "square root of x" -> "sqrt(x)"
        expression = re.sub(r'derivative\s+of\s+(.+)', r'diff(\1, x)', expression)  # "derivative of x**2" -> "diff(x**2, x)"
        expression = re.sub(r'integral\s+of\s+(.+)', r'integrate(\1, x)', expression)  # "integral of x**2" -> "integrate(x**2, x)"
        
        # Use sympy to safely evaluate the expression
        result = sympify(expression)
        
        # Simplify the result
        simplified_result = simplify(result)
        
        # Return clean, TTS-friendly response
        if hasattr(simplified_result, 'evalf'):
            # For simple numbers, return just the number
            numerical_result = simplified_result.evalf()
            if simplified_result.is_number and abs(float(numerical_result) - round(float(numerical_result))) < 0.0001:
                # It's a whole number or very close to one
                return str(int(round(float(numerical_result))))
            elif simplified_result.is_number:
                # It's a decimal number
                return str(float(numerical_result))
            else:
                # It's a symbolic expression
                return str(simplified_result)
        else:
            return str(simplified_result)
            
    except ImportError:
        # Fallback to basic eval for simple arithmetic if sympy is not available
        try:
            # Only allow basic arithmetic operations for safety
            allowed_chars = set('0123456789+-*/().')
            if all(c in allowed_chars or c.isspace() for c in expression):
                result = eval(expression)
                # Return clean result
                if isinstance(result, float) and result.is_integer():
                    return str(int(result))
                else:
                    return str(result)
            else:
                return "Error: Complex mathematical expressions require sympy library"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


def route_llm_output(llm_output: str) -> str:
    """
    Route LLM response to the correct tool if it's a function call, else return the text.
    Expects LLM output in JSON format like {'function': ..., 'arguments': {...}}.
    
    Args:
        llm_output (str): The raw output from the LLM
        
    Returns:
        str: Either the result of a function call or the original text
    """
    try:
        # Try to parse as JSON function call
        output = json.loads(llm_output.strip())
        
        if not isinstance(output, dict) or 'function' not in output:
            # Not a function call, return original text
            return llm_output
            
        func_name = output.get("function")
        args = output.get("arguments", {})
        
        print(f"[INFO] Function call detected: {func_name} with args: {args}")
        
        if func_name == "search_arxiv":
            query = args.get("query", "")
            if not query:
                return "Error: No query provided for arXiv search"
            return search_arxiv(query)
            
        elif func_name == "calculate":
            expr = args.get("expression", "")
            if not expr:
                return "Error: No expression provided for calculation"
            return calculate(expr)
            
        else:
            return f"Error: Unknown function '{func_name}'"
            
    except (json.JSONDecodeError, TypeError):
        # Not a JSON function call; return the text directly
        return llm_output
    except Exception as e:
        print(f"[ERROR] Error in function routing: {e}")
        return f"Error processing request: {str(e)}"


# Test functions for debugging
if __name__ == "__main__":
    # Test calculate function
    print("Testing calculate function:")
    print(calculate("2 + 2"))
    print(calculate("derivative of x**2"))
    print(calculate("sqrt(16)"))
    
    # Test search_arxiv function
    print("\nTesting search_arxiv function:")
    print(search_arxiv("quantum entanglement"))
    print(search_arxiv("machine learning"))
    
    # Test route_llm_output function
    print("\nTesting route_llm_output function:")
    function_call = '{"function": "calculate", "arguments": {"expression": "2+2"}}'
    print(route_llm_output(function_call))
    
    normal_text = "Hello, how can I help you today?"
    print(route_llm_output(normal_text))