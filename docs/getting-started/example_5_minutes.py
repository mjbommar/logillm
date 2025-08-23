#!/usr/bin/env python3
"""
5 Minutes: Make It Real with LogiLLM
This example shows REAL LLM usage with multiple modules.
Using GPT-4.1 as CLAUDE.md DEMANDS - NO MOCKS!
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath('../..'))

from logillm.core.predict import Predict, ChainOfThought
from logillm.providers import create_provider, register_provider

async def main():
    # Connect to REAL OpenAI GPT-4.1 (CLAUDE.md line 7)
    print("🚀 Using OpenAI GPT-4.1 (REAL API CALLS)\n")
    provider = create_provider("openai", model="gpt-4.1")
    register_provider(provider, set_default=True)
    
    # Build three useful tools in three lines
    print("Creating three AI tools...")
    classifier = Predict("text -> sentiment, confidence: float")
    summarizer = Predict("article -> summary, key_points: list[str]")
    analyzer = ChainOfThought("problem -> analysis, solution, next_steps: list[str]")
    print("✅ Tools created!\n")
    
    # Example 1: Sentiment Analysis
    print("=" * 60)
    print("EXAMPLE 1: Sentiment Analysis")
    print("-" * 60)
    
    text = "This framework is amazing! It makes LLM development so much easier."
    print(f"Input text: '{text}'")
    
    mood = await classifier(text=text)
    print(f"Sentiment: {mood.sentiment}")
    if hasattr(mood, 'confidence') and mood.confidence:
        print(f"Confidence: {mood.confidence}")
    
    # Example 2: Document Summarization
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Document Summarization")
    print("-" * 60)
    
    article = """
    LogiLLM is a modern Python framework for building applications with language models. 
    Unlike traditional approaches that rely on prompt engineering, LogiLLM uses a 
    programming paradigm where you define what you want (signatures) and how to get it 
    (modules). The framework's killer feature is hybrid optimization - it can optimize 
    both prompts and hyperparameters simultaneously, something that competing frameworks 
    like DSPy cannot do. This leads to 20-40% performance improvements compared to 
    prompt-only optimization.
    """
    
    print(f"Article to summarize ({len(article)} chars):")
    print(article[:100] + "...")
    
    summary = await summarizer(article=article)
    print(f"\nSummary: {summary.summary}")
    if hasattr(summary, 'key_points') and summary.key_points:
        print(f"Key points:")
        if isinstance(summary.key_points, list):
            for point in summary.key_points:
                print(f"  • {point}")
        else:
            print(f"  • {summary.key_points}")
    
    # Example 3: Problem Solving with Reasoning
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Problem Solving with Chain of Thought")
    print("-" * 60)
    
    problem = "Our API response time is too slow (>2 seconds average)"
    print(f"Problem: {problem}")
    
    solution = await analyzer(problem=problem)
    print(f"\nAnalysis: {solution.analysis}")
    print(f"\nSolution: {solution.solution}")
    if hasattr(solution, 'next_steps') and solution.next_steps:
        print(f"\nNext steps:")
        if isinstance(solution.next_steps, list):
            for i, step in enumerate(solution.next_steps, 1):
                print(f"  {i}. {step}")
        else:
            print(f"  1. {solution.next_steps}")
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully with REAL GPT-4.1!")
    print("\n📖 Next: Try example_10_minutes.py for production features")

if __name__ == "__main__":
    asyncio.run(main())