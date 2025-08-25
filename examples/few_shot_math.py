#!/usr/bin/env python3
"""Few-Shot Learning for Math Problems with LogiLLM.

This example demonstrates how few-shot learning dramatically improves 
performance on grade school math problems by providing step-by-step
reasoning examples.

Math problems are an excellent test case because:
1. Without examples, models often give wrong answers
2. With a few examples showing reasoning steps, accuracy jumps significantly
3. The improvement is clear and measurable

Prerequisites:
- OpenAI API key: export OPENAI_API_KEY=your_key
- Install LogiLLM with OpenAI support: pip install logillm[openai]
"""

import asyncio
import os
import re

from logillm.core.optimizers import Metric
from logillm.core.predict import Predict
from logillm.core.signatures import Signature, InputField, OutputField
from logillm.optimizers import BootstrapFewShot
from logillm.providers import create_provider, register_provider


def extract_number(text: str) -> float:
    """Extract the numerical answer from text."""
    try:
        # Look for numbers in the text (including decimals)
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            # Return the last number found (usually the answer)
            return float(numbers[-1])
    except:
        pass
    return -1


class MathMetric(Metric):
    """Metric for evaluating math answers."""
    
    def __call__(self, prediction: dict, reference: dict, **kwargs) -> float:
        """Check if the numerical answer matches."""
        pred_answer = extract_number(prediction.get("answer", ""))
        ref_answer = extract_number(reference.get("answer", ""))
        # Allow small floating point differences
        return 1.0 if abs(pred_answer - ref_answer) < 0.01 else 0.0
    
    def name(self) -> str:
        return "math_accuracy"


async def main():
    """Demonstrate few-shot learning on math problems."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY=your_key")
        return

    print("=== Few-Shot Learning for Math Problems ===")
    print("=" * 44)

    try:
        # Step 1: Set up provider
        provider = create_provider("openai", model="gpt-4.1")
        register_provider(provider, set_default=True)

        # Step 2: Create a signature for math problems
        class MathSignature(Signature):
            """Solve grade school math word problems.
            
            Show your work step by step, then provide the numerical answer.
            """
            question: str = InputField(desc="Math word problem")
            reasoning: str = OutputField(desc="Step-by-step solution")
            answer: str = OutputField(desc="Final numerical answer")

        # Create module
        math_solver = Predict(MathSignature, provider=provider)

        # Step 3: Test problems (multi-step reasoning required)
        test_problems = [
            {
                "question": "A farmer has 47 chickens. He buys 23 more, then sells 1/5 of his total. How many chickens does he have left?",
                "answer": "56"  # 47+23=70, 70/5=14, 70-14=56
            },
            {
                "question": "Jane had some money. She spent $45 on books and $28 on lunch. She has $67 left. How much money did she start with?",
                "answer": "140"  # 45+28+67=140
            },
            {
                "question": "A train has 8 cars. Each car has 12 rows with 4 seats per row. If 287 seats are occupied, how many are empty?",
                "answer": "97"  # 8*12*4=384, 384-287=97
            },
            {
                "question": "A recipe needs 3/4 cup of sugar. To make 6 batches, how many cups of sugar are needed?",
                "answer": "4.5"  # 3/4 * 6 = 4.5
            },
            {
                "question": "Students collected 342 cans on Monday, 189 on Tuesday, and twice Tuesday's amount on Wednesday. What's the total?",
                "answer": "909"  # 342 + 189 + (189*2) = 342 + 189 + 378 = 909
            },
        ]

        # Step 4: Test baseline performance (no examples)
        print("\n🧮 Testing WITHOUT examples:")
        print("-" * 30)
        
        baseline_correct = 0
        for prob in test_problems:
            result = await math_solver(question=prob["question"])
            predicted_answer = extract_number(result.outputs.get("answer", ""))
            expected_answer = extract_number(prob["answer"])
            
            is_correct = predicted_answer == expected_answer
            if is_correct:
                baseline_correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Q: {prob['question'][:50]}...")
            print(f"  → Got: {predicted_answer}, Expected: {expected_answer}")
            
        baseline_accuracy = baseline_correct / len(test_problems)
        print(f"\n📊 Baseline Accuracy: {baseline_accuracy:.1%}")

        # Step 5: Prepare training data
        print("\n📚 Preparing training examples...")
        
        training_data = [
            {
                "inputs": {"question": "Lisa has 18 marbles. She gives away 6. How many does she have left?"},
                "outputs": {
                    "reasoning": "Lisa starts with 18 marbles. She gives away 6 marbles. 18 - 6 = 12",
                    "answer": "12"
                }
            },
            {
                "inputs": {"question": "A box contains 5 red balls and 3 blue balls. How many balls in total?"},
                "outputs": {
                    "reasoning": "Red balls: 5, Blue balls: 3. Total = 5 + 3 = 8",
                    "answer": "8"
                }
            },
            {
                "inputs": {"question": "Mike works 8 hours per day for 5 days. How many hours does he work in total?"},
                "outputs": {
                    "reasoning": "Hours per day: 8, Number of days: 5. Total hours = 8 × 5 = 40",
                    "answer": "40"
                }
            },
            {
                "inputs": {"question": "A cake is divided into 12 pieces. If 7 pieces are eaten, how many remain?"},
                "outputs": {
                    "reasoning": "Total pieces: 12, Pieces eaten: 7. Remaining = 12 - 7 = 5",
                    "answer": "5"
                }
            },
            {
                "inputs": {"question": "Books cost $4 each. How much do 9 books cost?"},
                "outputs": {
                    "reasoning": "Cost per book: $4, Number of books: 9. Total cost = 4 × 9 = 36",
                    "answer": "36"
                }
            },
            {
                "inputs": {"question": "Anna saves $5 per week. How much will she save in 8 weeks?"},
                "outputs": {
                    "reasoning": "Savings per week: $5, Number of weeks: 8. Total savings = 5 × 8 = 40",
                    "answer": "40"
                }
            },
        ]

        # Step 6: Use bootstrap few-shot to improve
        print("• Finding effective examples...")
        print("• Optimizing demonstrations...")
        
        metric = MathMetric()
        bootstrap = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=3,  # Use 3 examples
            max_rounds=1,
        )

        # Set up a quieter config for bootstrapping
        bootstrap_config = bootstrap.config
        bootstrap_config.rescue_mode_threshold = 0.3  # Only rescue if really bad
        
        result = await bootstrap.optimize(
            module=math_solver,
            dataset=training_data,
        )
        
        improved_solver = result.optimized_module

        # Step 7: Test with few-shot examples
        print("\n🧮 Testing WITH examples:")
        print("-" * 30)
        
        improved_correct = 0
        for prob in test_problems:
            result = await improved_solver(question=prob["question"])
            predicted_answer = extract_number(result.outputs.get("answer", ""))
            expected_answer = extract_number(prob["answer"])
            
            is_correct = predicted_answer == expected_answer
            if is_correct:
                improved_correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Q: {prob['question'][:50]}...")
            if not is_correct:
                print(f"  Reasoning: {result.outputs.get('reasoning', '')[:100]}...")
            print(f"  → Got: {predicted_answer}, Expected: {expected_answer}")
            
        improved_accuracy = improved_correct / len(test_problems)
        improvement = improved_accuracy - baseline_accuracy

        # Step 8: Show results
        print("\n" + "=" * 44)
        print("📊 RESULTS SUMMARY")
        print("=" * 44)
        print(f"Without examples: {baseline_accuracy:.1%} ({baseline_correct}/{len(test_problems)} correct)")
        print(f"With examples:    {improved_accuracy:.1%} ({improved_correct}/{len(test_problems)} correct)")
        print(f"Improvement:      {improvement:+.1%}")
        
        # Show what examples were selected
        if hasattr(improved_solver, "demo_manager") and improved_solver.demo_manager:
            examples = improved_solver.demo_manager.demos
            print(f"\n📚 Selected Examples ({len(examples)} used):")
            for i, demo in enumerate(examples[:3], 1):
                q = demo.inputs.get("question", "")[:60]
                a = demo.outputs.get("answer", "")
                print(f"  {i}. Q: {q}... → {a}")
        
        print("\n✅ Key Insights:")
        print("• Math problems benefit greatly from step-by-step examples")
        print("• Few-shot learning teaches the model HOW to solve problems")
        print("• Even 2-3 good examples can dramatically improve accuracy")
        print("• Bootstrap optimization finds the most helpful examples")

    except ImportError:
        print("OpenAI provider not installed. Run:")
        print("pip install logillm[openai]")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())