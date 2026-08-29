"""Source contracts for Workshop's semantic theme boundary."""

from __future__ import annotations

import re
from pathlib import Path

from kai.workshop.appearance_preferences import WORKSHOP_APPEARANCE_THEMES

_ROOT = Path(__file__).parents[1]
_COMPONENT_CSS = _ROOT / "workshop-client" / "src" / "styles.css"
_THEME_CSS = _ROOT / "workshop-client" / "src" / "themes.css"
_THEME_TYPESCRIPT = _ROOT / "workshop-client" / "src" / "theme.ts"
_THEME_SOURCES = _ROOT / "home" / "docs" / "specs" / "workshop-theme-catalog.md"
_THEME_SCHEMES = {
    "atom-one-dark": "dark",
    "atom-one-light": "light",
    "dracula": "dark",
    "nord": "dark",
    "solarized-dark": "dark",
    "solarized-light": "light",
    "catppuccin-mocha": "dark",
    "catppuccin-latte": "light",
    "github-light-default": "light",
    "github-dark-default": "dark",
    "github-dark-dimmed": "dark",
}
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
_NORMAL_TEXT_TOKENS = (
    "--color-text",
    "--color-text-strong",
    "--color-text-muted",
    "--color-danger",
    "--color-success",
    "--color-warning",
    "--color-accent",
    "--color-special",
    "--color-info",
    "--color-code-inline",
)
_TEXT_BACKGROUNDS = ("--color-canvas", "--color-panel")


def _declarations(css: str) -> set[str]:
    return set(re.findall(r"(--[a-z0-9-]+)\s*:", css))


def _references(css: str) -> set[str]:
    return set(re.findall(r"var\((--[a-z0-9-]+)", css))


def _theme_blocks(css: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", css):
        selector = re.search(r'data-workshop-theme="([a-z0-9-]+)"', match.group("selectors"))
        if selector is None:
            continue
        declarations = {
            name: value.strip() for name, value in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", match.group("body"))
        }
        scheme = re.search(r"color-scheme\s*:\s*(dark|light)\s*;", match.group("body"))
        assert scheme is not None
        declarations["color-scheme"] = scheme.group(1)
        blocks[selector.group(1)] = declarations
    return blocks


def _rgb(value: str) -> tuple[int, int, int]:
    assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), value
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


def _luminance(value: str) -> float:
    channels = []
    for channel in _rgb(value):
        normalized = channel / 255
        channels.append(normalized / 12.92 if normalized <= 0.04045 else ((normalized + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(first: str, second: str) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_catalogs_and_css_define_the_same_complete_theme_set() -> None:
    theme_css = _THEME_CSS.read_text()
    theme_typescript = _THEME_TYPESCRIPT.read_text()
    blocks = _theme_blocks(theme_css)

    assert set(blocks) == set(_THEME_SCHEMES)
    assert {item.theme_id: item.color_scheme for item in WORKSHOP_APPEARANCE_THEMES} == _THEME_SCHEMES
    for theme_id, scheme in _THEME_SCHEMES.items():
        assert blocks[theme_id]["color-scheme"] == scheme
        assert set(blocks[theme_id]) >= _REQUIRED_THEME_TOKENS | {"color-scheme"}
        if theme_id != "atom-one-dark":
            assert f'themeId: "{theme_id}"' in theme_typescript
    signatures = {
        tuple(blocks[theme_id][token] for token in sorted(_REQUIRED_THEME_TOKENS)) for theme_id in _THEME_SCHEMES
    }
    assert len(signatures) == len(_THEME_SCHEMES)


def test_every_theme_meets_workshop_contrast_targets() -> None:
    failures: list[str] = []
    for theme_id, declarations in _theme_blocks(_THEME_CSS.read_text()).items():
        for foreground in _NORMAL_TEXT_TOKENS:
            for background in _TEXT_BACKGROUNDS:
                ratio = _contrast(declarations[foreground], declarations[background])
                if ratio < 4.5:
                    failures.append(f"{theme_id}: {foreground} on {background} is {ratio:.2f}:1")

        primary_ratio = _contrast(declarations["--color-canvas-inset"], declarations["--color-accent"])
        if primary_ratio < 4.5:
            failures.append(f"{theme_id}: primary control text is {primary_ratio:.2f}:1")

        for background in ("--color-canvas", "--color-surface"):
            focus_ratio = _contrast(declarations["--color-accent"], declarations[background])
            if focus_ratio < 3:
                failures.append(f"{theme_id}: focus on {background} is {focus_ratio:.2f}:1")
    assert failures == []


def test_component_styles_are_palette_neutral_and_fully_defined() -> None:
    component_css = _COMPONENT_CSS.read_text()
    defined = _declarations(_THEME_CSS.read_text())

    assert "--atom-" not in component_css
    assert re.search(r"#[0-9a-f]{3,8}\b", component_css, re.IGNORECASE) is None
    assert re.search(r"\brgba?\(", component_css, re.IGNORECASE) is None
    semantic_references = {token for token in _references(component_css) if token.startswith(("--color-", "--shadow-"))}
    assert semantic_references <= defined
    assert "accent-color: var(--color-accent)" in component_css
    assert "scrollbar-color: var(--color-text-muted) var(--color-canvas-inset)" in component_css
    assert "::selection" in component_css
    assert "outline: 2px solid var(--color-accent)" in component_css


def test_component_styles_use_readable_scale_without_hiding_context() -> None:
    component_css = _COMPONENT_CSS.read_text()

    assert re.search(r":root\s*\{[^}]*font-size:\s*150%;", component_css, re.DOTALL)
    assert re.search(r"\.context-pane\s*\{\s*display:\s*none", component_css) is None


def test_theme_sources_are_attributed_at_pinned_revisions() -> None:
    attribution = _THEME_SOURCES.read_text()
    for project in (
        "atom/one-dark-syntax",
        "atom/one-light-syntax",
        "dracula/dracula-theme",
        "nordtheme/nord",
        "altercation/solarized",
        "catppuccin/palette",
        "primer/github-vscode-theme",
        "primer/primitives",
    ):
        assert project in attribution
    assert attribution.count("MIT") >= 7
    assert len(re.findall(r"`[0-9a-f]{40}`", attribution)) >= 8
