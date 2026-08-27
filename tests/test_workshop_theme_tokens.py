"""Source contracts for Workshop's semantic theme boundary."""

import re
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_COMPONENT_CSS = _ROOT / "workshop-client" / "src" / "styles.css"
_THEME_CSS = _ROOT / "workshop-client" / "src" / "themes.css"
_REQUIRED_THEME_TOKENS = frozenset(
    {
        "--color-canvas",
        "--color-panel",
        "--color-canvas-inset",
        "--color-surface",
        "--color-surface-raised",
        "--color-text",
        "--color-text-strong",
        "--color-text-muted",
        "--color-danger",
        "--color-success",
        "--color-warning",
        "--color-accent",
        "--color-accent-hover",
        "--color-special",
        "--color-info",
        "--color-code-inline",
        "--color-border",
        "--color-border-strong",
        "--color-shadow-base",
        "--color-backdrop",
        "--color-overlay-shadow",
        "--shadow-elevation-page",
    }
)


def _declarations(css: str) -> set[str]:
    return set(re.findall(r"(--[a-z0-9-]+)\s*:", css))


def _references(css: str) -> set[str]:
    return set(re.findall(r"var\((--[a-z0-9-]+)", css))


def test_atom_one_dark_defines_complete_semantic_theme_contract() -> None:
    theme_css = _THEME_CSS.read_text()

    assert ':root[data-workshop-theme="atom-one-dark"]' in theme_css
    assert _declarations(theme_css) >= _REQUIRED_THEME_TOKENS


def test_component_styles_are_palette_neutral_and_fully_defined() -> None:
    component_css = _COMPONENT_CSS.read_text()
    defined = _declarations(_THEME_CSS.read_text())

    assert "--atom-" not in component_css
    assert re.search(r"#[0-9a-f]{3,8}\b", component_css, re.IGNORECASE) is None
    assert re.search(r"\brgba?\(", component_css, re.IGNORECASE) is None
    semantic_references = {token for token in _references(component_css) if token.startswith(("--color-", "--shadow-"))}
    assert semantic_references <= defined
