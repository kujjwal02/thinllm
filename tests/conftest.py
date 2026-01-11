"""Pytest configuration and fixtures for thinllm tests."""

import os

import pytest
from dotenv import load_dotenv

# The location of my CA File
cert_file = os.path.expanduser("~/.mitmproxy/mitmproxy-ca-cert.pem")
os.environ["REQUESTS_CA_BUNDLE"] = cert_file
os.environ["SSL_CERT_FILE"] = cert_file
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8080"


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """
    Load OpenAI API key from .env file or environment variable.

    This fixture:
    1. First tries to load from .env file using python-dotenv
    2. Falls back to existing environment variable
    3. Skips the test if API key is not found in either location

    Returns:
        str: The OpenAI API key

    Raises:
        pytest.skip: If OPENAI_API_KEY is not found (skips only tests using this fixture)
    """
    # Try to load from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        pytest.skip(
            "OPENAI_API_KEY not found. Please set it in .env file or as an environment variable."
        )

    return api_key


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """
    Load Anthropic API key from .env file or environment variable.

    This fixture:
    1. First tries to load from .env file using python-dotenv
    2. Falls back to existing environment variable
    3. Skips the test if API key is not found in either location

    Returns:
        str: The Anthropic API key

    Raises:
        pytest.skip: If ANTHROPIC_API_KEY is not found (skips only tests using this fixture)
    """
    # Try to load from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        pytest.skip(
            "ANTHROPIC_API_KEY not found. Please set it in .env file or as an environment variable."
        )

    return api_key


@pytest.fixture(scope="session")
def gemini_api_key() -> str:
    """
    Load Gemini API key from .env file or environment variable.

    This fixture:
    1. First tries to load from .env file using python-dotenv
    2. Falls back to existing environment variable
    3. Skips the test if API key is not found in either location

    Returns:
        str: The Gemini API key

    Raises:
        pytest.skip: If GEMINI_API_KEY is not found (skips only tests using this fixture)
    """
    # Try to load from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        pytest.skip(
            "GEMINI_API_KEY not found. Please set it in .env file or as an environment variable."
        )

    return api_key
