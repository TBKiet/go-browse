"""
Embedding utilities for frontier diversity scoring.

Provides functions to generate vector embeddings from webpage screenshots
or DOM text, used to compute the Diversity (D) component of frontier scoring.
"""

from __future__ import annotations
from typing import List, Optional
import logging
import os
import base64
import io

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Create an OpenAI client, falling back to environment variables."""
    from openai import OpenAI

    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
    )


def compute_screenshot_embedding(
    screenshot,
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[List[float]]:
    """Generate an embedding vector from a screenshot using a vision-capable
    embedding model or an LLM that can describe the screenshot.

    This is a two-step approach:
    1. Use a vision model to describe the screenshot in a structured caption.
    2. Use a text embedding model on that caption.

    Args:
        screenshot: numpy array (H, W, C) from the browser env observation.
        model_name: Text embedding model name.
        api_key: API key. Falls back to OPENAI_API_KEY env var.
        base_url: Custom API base URL. Falls back to OPENAI_BASE_URL env var.

    Returns:
        List of floats (embedding vector), or None on failure.
    """
    try:
        client = _get_openai_client(api_key, base_url)

        # Step 1: Describe screenshot → text caption
        # Convert numpy array to base64 PNG
        from PIL import Image
        img = Image.fromarray(screenshot)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        caption_resp = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast, cheap vision model
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Describe the visual layout and key UI elements of this webpage "
                        "in 1-2 sentences. Focus on structure (e.g., 'search bar at top, "
                        "product grid below, footer with links')."
                    )},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=100,
        )
        caption = caption_resp.choices[0].message.content.strip()
        logger.debug(f"Screenshot caption: {caption[:100]}...")

        # Step 2: Embed the caption
        emb_resp = client.embeddings.create(
            model=model_name,
            input=caption,
        )
        return emb_resp.data[0].embedding

    except Exception as e:
        logger.warning(f"Failed to compute screenshot embedding: {e}")
        return None


def compute_dom_embedding(
    dom_text: str,
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[List[float]]:
    """Generate an embedding vector from DOM text content.

    Args:
        dom_text: Textual representation of the page DOM.
        model_name: Embedding model name.
        api_key: API key. Falls back to OPENAI_API_KEY env var.
        base_url: Custom API base URL. Falls back to OPENAI_BASE_URL env var.

    Returns:
        List of floats (embedding vector), or None on failure.
    """
    try:
        client = _get_openai_client(api_key, base_url)

        # Truncate to avoid token limits
        max_chars = 8000
        truncated = dom_text[:max_chars]

        emb_resp = client.embeddings.create(
            model=model_name,
            input=truncated,
        )
        return emb_resp.data[0].embedding

    except Exception as e:
        logger.warning(f"Failed to compute DOM embedding: {e}")
        return None
