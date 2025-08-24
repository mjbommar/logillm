"""
Interactive learning session for code generation tutorial.
"""

from typing import Any

from .generator import LibraryCodeGenerator


class InteractiveLearningSession:
    """Interactive session for learning new libraries."""

    def __init__(self) -> None:
        self.generator = LibraryCodeGenerator()
        self.current_library: str = ""
        self.current_analysis: Any = None

    def display_analysis(self, analysis: Any) -> None:
        """Display library analysis in a user-friendly format."""
        print("\n📊 LIBRARY ANALYSIS")
        print("=" * 50)
        print(f"📦 Installation: {analysis.installation_method}")

        print(f"\n🔑 Core Concepts ({len(analysis.core_concepts)}):")
        for i, concept in enumerate(analysis.core_concepts, 1):
            print(f"   {i}. {concept}")

        print(f"\n🏗️  Main Classes ({len(analysis.main_classes)}):")
        for i, cls in enumerate(analysis.main_classes, 1):
            print(f"   {i}. {cls}")

        print(f"\n⚡ Common Patterns ({len(analysis.common_patterns)}):")
        for i, pattern in enumerate(analysis.common_patterns, 1):
            print(f"   {i}. {pattern}")

        print("\n📥 Import Statements:")
        for stmt in analysis.import_statements:
            print(f"   {stmt}")

    def display_code_example(self, use_case: str, example: Any) -> None:
        """Display a code example with explanation."""
        print(f"\n💻 CODE EXAMPLE: {use_case}")
        print("=" * 50)
        print(example.code_example)

        print("\n📝 EXPLANATION:")
        print(example.explanation)

        if example.best_practices:
            print("\n✅ BEST PRACTICES:")
            for i, practice in enumerate(example.best_practices, 1):
                print(f"   {i}. {practice}")

        if example.potential_issues:
            print("\n⚠️  POTENTIAL ISSUES:")
            for i, issue in enumerate(example.potential_issues, 1):
                print(f"   {i}. {issue}")

    def display_learning_plan(self, plan: Any) -> None:
        """Display the learning plan."""
        print("\n🎯 LEARNING PLAN")
        print("=" * 50)
        print(f"⏱️  Estimated Time: {plan.time_estimate}")

        print("\n🎯 Learning Objectives:")
        for i, obj in enumerate(plan.learning_objectives, 1):
            print(f"   {i}. {obj}")

        print("\n📚 Suggested Examples (in order):")
        for i, example in enumerate(plan.suggested_examples, 1):
            print(f"   {i}. {example}")

        if plan.additional_resources:
            print("\n🔗 Additional Resources:")
            for resource in plan.additional_resources:
                print(f"   • {resource}")

    async def explore_library(self, library_name: str, user_experience: str = "beginner") -> None:
        """Start an interactive exploration of a library."""

        self.current_library = library_name

        # Common use cases for different types of libraries
        common_use_cases = [
            "basic setup and hello world",
            "intermediate usage with common features",
            "advanced usage with best practices",
            "error handling and debugging",
        ]

        try:
            result = await self.generator.forward(
                library_name=library_name,
                use_cases=common_use_cases,
                user_experience=user_experience,
                fetch_docs=False,  # Use fallback for reliability
            )

            self.current_analysis = result.analysis

            # Display results
            self.display_analysis(result.analysis)

            print(f"\n🎓 Generated {len(result.code_examples)} code examples")

            # Show each example
            for use_case, example in result.code_examples.items():
                self.display_code_example(use_case, example)

                # Wait for user input to continue
                input("\nPress Enter to continue to the next example...")

            # Show learning plan
            if result.learning_plan:
                self.display_learning_plan(result.learning_plan)

            print(f"\n🎉 Exploration of {library_name} complete!")

        except Exception as e:
            print(f"❌ Error exploring library: {e}")
            import traceback

            traceback.print_exc()

    async def refine_example(self, use_case: str, feedback: str) -> None:
        """Refine a previously generated example based on feedback."""
        if not self.current_analysis:
            print("❌ No current analysis available. Explore a library first.")
            return

        print(f"🔄 Refining example for '{use_case}' based on feedback...")
        print(
            "(This would refine a previous example - implementation depends on storing previous results)"
        )

        # In a full implementation, you would:
        # 1. Retrieve the previous example for the use case
        # 2. Use the refine_code method from the generator
        # 3. Display the improved result
