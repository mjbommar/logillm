"""
Repository analyzer module using LogiLLM.
"""

from typing import Any

from logillm.core.modules import Module
from logillm.core.predict import Predict

from .signatures import (
    AnalyzeCodeStructure,
    AnalyzeRepository,
    GenerateLLMsTxt,
    GenerateUsageExamples,
)


class RepositoryAnalyzer(Module):
    """A comprehensive repository documentation generator using LogiLLM."""

    def __init__(self) -> None:
        super().__init__()

        # Initialize our analysis components
        self.analyze_repo = Predict(signature=AnalyzeRepository)
        self.analyze_structure = Predict(signature=AnalyzeCodeStructure)
        self.generate_examples = Predict(signature=GenerateUsageExamples)
        self.generate_llms_txt = Predict(signature=GenerateLLMsTxt)

    async def forward(
        self, repo_url: str, file_tree: str, readme_content: str, package_files: str
    ) -> dict[str, Any]:
        """Process repository information and generate llms.txt documentation."""

        # Step 1: Analyze repository purpose and concepts
        repo_analysis = await self.analyze_repo(
            repo_url=repo_url, file_tree=file_tree, readme_content=readme_content
        )

        # Step 2: Analyze code structure
        structure_analysis = await self.analyze_structure(
            file_tree=file_tree, package_files=package_files
        )

        # Step 3: Generate practical usage examples
        usage_examples = await self.generate_examples(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
        )

        # Step 4: Generate final llms.txt
        llms_txt = await self.generate_llms_txt(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
            important_directories=structure_analysis.important_directories,
            entry_points=structure_analysis.entry_points,
            development_info=structure_analysis.development_info,
            usage_examples=usage_examples.usage_examples,
        )

        return {
            "llms_txt_content": llms_txt.llms_txt_content,
            "analysis": repo_analysis,
            "structure": structure_analysis,
            "examples": usage_examples,
        }
