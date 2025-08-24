"""
LogiLLM signatures for repository documentation generation.
"""

from logillm.core.signatures import InputField, OutputField, Signature


class AnalyzeRepository(Signature):
    """Analyze a repository structure and identify key components."""

    repo_url: str = InputField(desc="GitHub repository URL")
    file_tree: str = InputField(desc="Repository file structure as text")
    readme_content: str = InputField(desc="README.md content")

    project_purpose: str = OutputField(desc="Main purpose and goals of the project")
    key_concepts: list[str] = OutputField(desc="Important concepts and terminology")
    architecture_overview: str = OutputField(desc="High-level architecture description")


class AnalyzeCodeStructure(Signature):
    """Analyze code structure to identify important directories and files."""

    file_tree: str = InputField(desc="Repository file structure")
    package_files: str = InputField(desc="Key package and configuration files")

    important_directories: list[str] = OutputField(desc="Key directories and their purposes")
    entry_points: list[str] = OutputField(desc="Main entry points and important files")
    development_info: str = OutputField(desc="Development setup and workflow information")


class GenerateUsageExamples(Signature):
    """Generate practical usage examples based on project information."""

    project_purpose: str = InputField()
    key_concepts: list[str] = InputField()
    architecture_overview: str = InputField()

    usage_examples: str = OutputField(desc="Practical usage examples and code snippets")


class GenerateLLMsTxt(Signature):
    """Generate a comprehensive llms.txt file from analyzed repository information."""

    project_purpose: str = InputField()
    key_concepts: list[str] = InputField()
    architecture_overview: str = InputField()
    important_directories: list[str] = InputField()
    entry_points: list[str] = InputField()
    development_info: str = InputField()
    usage_examples: str = InputField()

    llms_txt_content: str = OutputField(
        desc="Complete llms.txt file content following standard format"
    )
