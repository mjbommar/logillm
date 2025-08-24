"""
Web scraping and documentation processing utilities.
"""

import time
from typing import Optional

import requests
from bs4 import BeautifulSoup


class DocumentationFetcher:
    """Fetches and processes documentation from library websites."""

    def __init__(self, timeout: int = 30, retry_delay: float = 1.0) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "LogiLLM Code Generator Bot (Educational Use)"})
        self.timeout = timeout
        self.retry_delay = retry_delay

    def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch raw HTML content from a URL."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def extract_text_content(self, html_content: str) -> str:
        """Extract clean text content from HTML."""
        soup = BeautifulSoup(html_content, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    def fetch_documentation_urls(self, library_name: str) -> list[str]:
        """Generate potential documentation URLs for a library."""
        base_patterns = [
            f"https://{library_name}.readthedocs.io/",
            f"https://{library_name}.readthedocs.io/en/latest/",
            f"https://docs.{library_name}.org/",
            f"https://{library_name}.org/docs/",
            f"https://{library_name}.org/documentation/",
            f"https://github.com/{library_name}/{library_name}#readme",
            f"https://pypi.org/project/{library_name}/",
        ]
        return base_patterns

    def fetch_library_documentation(self, library_name: str) -> dict[str, str]:
        """Fetch documentation from multiple sources for a library."""
        urls = self.fetch_documentation_urls(library_name)
        docs = {}

        for url in urls:
            print(f"🔍 Fetching documentation from: {url}")
            html_content = self.fetch_page_content(url)

            if html_content:
                text_content = self.extract_text_content(html_content)
                if text_content and len(text_content) > 100:  # Minimum content threshold
                    docs[url] = text_content
                    print(f"✅ Successfully fetched {len(text_content)} characters")
                else:
                    print(f"⚠️  Insufficient content from {url}")
            else:
                print(f"❌ Failed to fetch content from {url}")

            # Be respectful with rate limiting
            time.sleep(self.retry_delay)

        return docs

    def combine_documentation(self, docs: dict[str, str]) -> str:
        """Combine documentation from multiple sources."""
        combined = []

        for url, content in docs.items():
            combined.append(f"=== Documentation from {url} ===\n")
            combined.append(content[:5000])  # Limit content per source
            combined.append("\n\n")

        return "".join(combined)


def create_common_library_examples() -> dict[str, dict[str, str]]:
    """Create examples for common libraries when web scraping isn't available."""
    return {
        "fastapi": {
            "basic_setup": """
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
            """,
            "documentation": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+",
        },
        "requests": {
            "basic_setup": """
import requests

# GET request
response = requests.get('https://httpbin.org/get')
print(response.json())

# POST request with data
data = {'key': 'value'}
response = requests.post('https://httpbin.org/post', json=data)
print(response.status_code)
            """,
            "documentation": "Requests is a simple, elegant HTTP library for Python",
        },
        "pandas": {
            "basic_setup": """
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)

# Basic operations
print(df.head())
print(df.describe())
            """,
            "documentation": "Pandas is a fast, powerful data analysis and manipulation library",
        },
        "click": {
            "basic_setup": """
import click

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def hello(count, name):
    for i in range(count):
        click.echo(f'Hello {name}!')

if __name__ == '__main__':
    hello()
            """,
            "documentation": "Click is a Python package for creating command line interfaces",
        },
    }
