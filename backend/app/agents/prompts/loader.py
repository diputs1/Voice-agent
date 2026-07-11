from __future__ import annotations

from app.prompts import Prompt, load_prompt as load_shared_prompt


def load_prompt(name: str) -> Prompt:
    """Load prompt templates from the shared app prompt store."""
    return load_shared_prompt(name)
