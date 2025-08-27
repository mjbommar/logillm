"""
Core code generation module using LogiLLM.
"""

import json
from typing import Any, Optional

from logillm.core.modules import Module
from logillm.core.predict import Predict

from .doc_fetcher import DocumentationFetcher, create_common_library_examples
from .signatures import (
    AnalyzeLibraryDocs,
    CreateLearningPlan,
    GenerateCodeExample,
    RefineCodeExample,
)


class CodeGenerationResult:
    """Container for code generation results."""

    def __init__(
        self,
        library_name: str,
        analysis: Any,
        code_examples: dict[str, Any],
        learning_plan: Optional[Any] = None,
    ):
        self.library_name = library_name
        self.analysis = analysis
        self.code_examples = code_examples
        self.learning_plan = learning_plan


class LibraryCodeGenerator(Module):
    """Main module for generating code examples for unfamiliar libraries."""

    def __init__(self) -> None:
        super().__init__()

        # Initialize LogiLLM components
        self.doc_analyzer = Predict(signature=AnalyzeLibraryDocs)
        self.code_generator = Predict(signature=GenerateCodeExample)
        self.learning_planner = Predict(signature=CreateLearningPlan)
        self.code_refiner = Predict(signature=RefineCodeExample)

        # Initialize documentation fetcher
        self.doc_fetcher = DocumentationFetcher()

        # Common library examples for fallback
        self.common_examples = create_common_library_examples()

    def _parse_json_list(self, text: str) -> list[str]:
        """Parse JSON array from text with fallbacks."""
        if not text:
            return []

        text = text.strip()

        # Try JSON array format
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Fallback: split by comma or newlines
        if "," in text:
            return [item.strip() for item in text.split(",")]
        elif "\n" in text:
            return [item.strip() for item in text.split("\n") if item.strip()]

        return [text] if text else []

    async def analyze_library(self, library_name: str, documentation: str) -> Any:
        """Analyze library documentation to extract key information."""
        return await self.doc_analyzer(
            library_name=library_name, documentation_content=documentation
        )

    async def generate_example(self, library_name: str, use_case: str, analysis: Any) -> Any:
        """Generate a code example for a specific use case."""
        return await self.code_generator(
            library_name=library_name,
            use_case=use_case,
            core_concepts=analysis.core_concepts,
            common_patterns=analysis.common_patterns,
            import_statements=analysis.import_statements,
        )

    async def create_learning_plan(
        self, library_name: str, analysis: Any, user_experience: str = "beginner"
    ) -> Any:
        """Create a structured learning plan for the library."""
        return await self.learning_planner(
            library_name=library_name,
            core_concepts=analysis.core_concepts,
            common_patterns=analysis.common_patterns,
            user_experience=user_experience,
        )

    async def refine_code(
        self, original_code: str, feedback: str, library_patterns: list[str] | str
    ) -> Any:
        """Refine and improve a code example based on feedback."""
        # Convert list to string if needed
        if isinstance(library_patterns, list):
            patterns_str = json.dumps(library_patterns)
        else:
            patterns_str = library_patterns

        return await self.code_refiner(
            original_code=original_code, feedback=feedback, library_patterns=patterns_str
        )

    async def forward(
        self,
        library_name: str,
        use_cases: list[str],
        user_experience: str = "beginner",
        fetch_docs: bool = True,
    ) -> CodeGenerationResult:
        """Generate comprehensive code examples for a library."""

        print(f"🚀 Generating code examples for {library_name}")

        # Step 1: Get documentation
        if fetch_docs and library_name not in self.common_examples:
            print("📚 Fetching documentation...")
            docs = self.doc_fetcher.fetch_library_documentation(library_name)
            documentation = self.doc_fetcher.combine_documentation(docs)
        else:
            # Use fallback documentation if available
            if library_name in self.common_examples:
                documentation = self.common_examples[library_name]["documentation"]
                print("📚 Using fallback documentation...")
            else:
                documentation = f"Limited documentation available for {library_name}"
                print("⚠️  Using minimal documentation...")

        print(f"📖 Documentation length: {len(documentation)} characters")

        # Step 2: Analyze the library
        print("🔍 Analyzing library structure...")
        analysis = await self.analyze_library(library_name, documentation)

        # Parse string lists for display
        parsed_concepts = self._parse_json_list(analysis.core_concepts)
        parsed_patterns = self._parse_json_list(analysis.common_patterns)

        print(f"✅ Found {len(parsed_concepts)} core concepts")
        print(f"✅ Identified {len(parsed_patterns)} patterns")

        # Step 3: Generate code examples for each use case
        print("💻 Generating code examples...")
        code_examples = {}

        for use_case in use_cases:
            print(f"   Generating example for: {use_case}")
            example = await self.generate_example(library_name, use_case, analysis)
            code_examples[use_case] = example

        # Step 4: Create learning plan
        print("📝 Creating learning plan...")
        learning_plan = await self.create_learning_plan(library_name, analysis, user_experience)

        return CodeGenerationResult(
            library_name=library_name,
            analysis=analysis,
            code_examples=code_examples,
            learning_plan=learning_plan,
        )
