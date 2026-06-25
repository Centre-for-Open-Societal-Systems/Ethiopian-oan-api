"""Jinja2 template loader for fast-path responses (EN + AM)."""
from __future__ import annotations
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import settings

TEMPLATES_DIR = settings.base_dir / "assets" / "prompts" / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(name: str, lang: str, **ctx) -> str:
    lang = lang if lang in ("en", "am") else "en"
    template_path = f"{lang}/{name}.jinja"
    template = _env.get_template(template_path)
    return template.render(**ctx).strip()
