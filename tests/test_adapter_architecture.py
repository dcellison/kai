"""Static dependency gates for optional client adapters."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "kai"

_TELEGRAM_SDK_ADAPTERS = frozenset(
    {
        "kai.bot",
        "kai.memory_command",
        "kai.telegram_adapter",
        "kai.telegram_context",
        "kai.telegram_http",
        "kai.workshop.telegram_delivery",
        "kai.workshop.telegram_delivery_runtime",
    }
)
_TELEGRAM_IMPLEMENTATION_MODULES = _TELEGRAM_SDK_ADAPTERS | {
    "kai.telegram_utils",
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT.parent).with_suffix("")
    return ".".join(relative.parts)


def _imports(path: Path) -> tuple[str, ...]:
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            source = node.module or ""
            if node.level:
                source = importlib.util.resolve_name(f"{'.' * node.level}{source}", package)
            if source:
                imported.append(source)
            imported.extend(f"{source}.{alias.name}" for alias in node.names if source and alias.name != "*")
    return tuple(imported)


def _is_module_or_child(imported: str, modules: frozenset[str]) -> bool:
    return any(imported == module or imported.startswith(f"{module}.") for module in modules)


def test_only_telegram_adapter_modules_import_the_optional_sdk() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        module = _module_name(path)
        for imported in _imports(path):
            if (imported == "telegram" or imported.startswith("telegram.")) and module not in _TELEGRAM_SDK_ADAPTERS:
                violations.append(f"{module} imports {imported}")
    assert violations == []


def test_core_and_feature_services_do_not_import_telegram_implementation() -> None:
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        module = _module_name(path)
        if module in _TELEGRAM_IMPLEMENTATION_MODULES:
            continue
        for imported in _imports(path):
            if _is_module_or_child(imported, _TELEGRAM_IMPLEMENTATION_MODULES):
                violations.append(f"{module} imports {imported}")
    assert violations == []


def test_composition_root_loads_telegram_adapter_only_when_enabled() -> None:
    main_source = (SOURCE_ROOT / "main.py").read_text()

    assert 'import_module("kai.telegram_adapter")' in main_source
    assert "from kai.telegram_adapter" not in main_source
    assert "import kai.telegram_adapter" not in main_source


def test_core_identity_and_cancellation_services_are_transport_generic() -> None:
    for relative in (
        "workshop/client_access.py",
        "workshop/private_text_execution.py",
        "workshop/storage_namespaces.py",
    ):
        source = (SOURCE_ROOT / relative).read_text()
        assert "telegram" not in source.lower(), relative


def test_shared_http_host_exports_no_telegram_application_state() -> None:
    source = (SOURCE_ROOT / "webhook.py").read_text()

    assert "TELEGRAM_APP_KEY" not in source
    assert "TELEGRAM_BOT_KEY" not in source
    assert "TELEGRAM_WEBHOOK_SECRET_KEY" not in source
    assert "NOTIFICATION_CHAT_IDS_KEY" not in source
    assert "CHAT_ID_KEY" not in source


def test_workshop_package_does_not_export_adapter_preview_types() -> None:
    source = (SOURCE_ROOT / "workshop" / "__init__.py").read_text()

    assert "streaming_preview" not in source
    assert "TelegramStreamingPreview" not in source
