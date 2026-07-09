from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: str

    def format(self, **values: object) -> str:
        return self.template.format(**values)


@lru_cache(maxsize=16)
def load_prompt(name: str) -> Prompt:
    path = Path(__file__).with_name(f"{name}.md")
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return _parse_prompt_text(name, path.read_text(encoding="utf-8"))


def _parse_prompt_text(name: str, raw: str) -> Prompt:
    raw = raw.strip()
    first_line, _, body = raw.partition("\n")
    prefix = "prompt_version: "
    if not first_line.startswith(prefix):
        raise ValueError(f"Prompt {name} is missing a prompt_version header")

    version = first_line.removeprefix(prefix).strip()
    if not version.startswith(f"{name}@"):
        raise ValueError(f"Prompt {name} version must start with {name}@")
    if not body.strip():
        raise ValueError(f"Prompt {name} body is empty")

    return Prompt(name=name, version=version, template=body.strip())
