"""
GitHub API utilities for repository analysis.
"""

import base64
from typing import Optional

import requests


def get_github_file_tree(repo_url: str, github_token: Optional[str] = None) -> str:
    """Get repository file structure from GitHub API."""

    # Extract owner/repo from URL
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid repository URL: {repo_url}")

    owner, repo = parts[-2], parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"

    headers = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    response = requests.get(api_url, headers=headers, timeout=30)

    if response.status_code == 200:
        tree_data = response.json()
        file_paths = [
            item["path"]
            for item in tree_data["tree"]
            if item["type"] == "blob" and not item["path"].startswith(".")
        ]
        return "\n".join(sorted(file_paths))
    else:
        raise Exception(f"Failed to fetch repository tree: {response.status_code}")


def get_github_file_content(
    repo_url: str, file_path: str, github_token: Optional[str] = None
) -> str:
    """Get specific file content from GitHub."""

    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid repository URL: {repo_url}")

    owner, repo = parts[-2], parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"

    headers = {}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    response = requests.get(api_url, headers=headers, timeout=30)

    if response.status_code == 200:
        content_b64 = response.json()["content"]
        content = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        return content
    else:
        return f"Could not fetch {file_path}"


def gather_repository_info(
    repo_url: str, github_token: Optional[str] = None
) -> tuple[str, str, str]:
    """Gather all necessary repository information."""

    # Get file tree
    file_tree = get_github_file_tree(repo_url, github_token)

    # Get README content
    readme_content = get_github_file_content(repo_url, "README.md", github_token)

    # Get key package files
    package_files = []
    for file_path in [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "package.json",
        "Cargo.toml",
    ]:
        try:
            content = get_github_file_content(repo_url, file_path, github_token)
            if "Could not fetch" not in content:
                package_files.append(f"=== {file_path} ===\n{content}")
        except Exception:
            continue

    package_files_content = (
        "\n\n".join(package_files) if package_files else "No package files found"
    )

    return file_tree, readme_content, package_files_content
