"""
Main application for generating llms.txt documentation using LogiLLM.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from logillm.providers import create_provider, register_provider

from .analyzer import RepositoryAnalyzer
from .github_utils import gather_repository_info


async def generate_llms_txt_for_repo(
    repo_url: str,
    output_path: str = "llms.txt",
    github_token: Optional[str] = None,
    model: str = "gpt-4.1",
) -> None:
    """Generate llms.txt documentation for a GitHub repository."""

    print(f"🔍 Analyzing repository: {repo_url}")

    # Step 1: Setup LogiLLM provider
    if model.startswith("gpt"):
        provider = create_provider("openai", model=model)
    elif model.startswith("claude"):
        provider = create_provider("anthropic", model=model)
    else:
        raise ValueError(f"Unsupported model: {model}")

    register_provider(provider, set_default=True)

    # Step 2: Initialize analyzer
    analyzer = RepositoryAnalyzer()

    # Step 3: Gather repository information
    print("📁 Gathering repository information...")
    file_tree, readme_content, package_files = gather_repository_info(repo_url, github_token)

    # Step 4: Generate documentation
    print("🤖 Generating llms.txt documentation...")
    result = await analyzer.forward(
        repo_url=repo_url,
        file_tree=file_tree,
        readme_content=readme_content,
        package_files=package_files,
    )

    # Step 5: Save the generated documentation
    output_file = Path(output_path)
    output_file.write_text(result["llms_txt_content"])

    print(f"✅ Generated llms.txt saved to: {output_file.absolute()}")
    print(f"📊 Project: {result['analysis'].project_purpose[:100]}...")
    print(f"🏗️  Architecture: {result['analysis'].architecture_overview[:100]}...")

    # Step 6: Show preview
    print("\n📄 Preview (first 500 characters):")
    print("-" * 50)
    content = result["llms_txt_content"]
    preview = content[:500] + "..." if len(content) > 500 else content
    print(preview)


async def main() -> None:
    """Main application entry point."""

    # Configuration
    repo_url = os.environ.get("REPO_URL", "https://github.com/octocat/Hello-World")
    github_token = os.environ.get("GITHUB_TOKEN")
    model = os.environ.get("MODEL", "gpt-4.1")
    output_path = os.environ.get("OUTPUT_PATH", "llms.txt")

    await generate_llms_txt_for_repo(
        repo_url=repo_url, github_token=github_token, model=model, output_path=output_path
    )


if __name__ == "__main__":
    asyncio.run(main())
