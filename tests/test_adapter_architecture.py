"""Static dependency gates for optional client adapters."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from kai.capability_boundaries import CANONICAL_SERVICE_BOUNDARIES
from kai.capability_registry import CAPABILITY_REGISTRY, AdapterId

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
_WORKSHOP_PRESENTATION_MODULES = frozenset(
    {
        "kai.http_adapter",
        "kai.webhook",
        "kai.workshop.client_access",
        "kai.workshop.client_api",
        "kai.workshop.client_sessions",
        "kai.workshop.client_shell",
        "kai.workshop.telegram_delivery",
        "kai.workshop.telegram_delivery_runtime",
    }
)
_ADAPTER_FRAMEWORK_MODULES = _TELEGRAM_IMPLEMENTATION_MODULES | _WORKSHOP_PRESENTATION_MODULES


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


def test_registered_canonical_service_modules_exist_and_are_transport_neutral() -> None:
    violations: list[str] = []
    registered_modules = {module for boundary in CANONICAL_SERVICE_BOUNDARIES.values() for module in boundary.modules}
    for module in sorted(registered_modules):
        path = SOURCE_ROOT.parent.joinpath(*module.split(".")).with_suffix(".py")
        if not path.is_file():
            violations.append(f"{module} is missing")
            continue
        for imported in _imports(path):
            if imported == "aiohttp" or imported.startswith("aiohttp."):
                violations.append(f"{module} imports Workshop HTTP type {imported}")
            if imported == "telegram" or imported.startswith("telegram."):
                violations.append(f"{module} imports Telegram framework type {imported}")
            if _is_module_or_child(imported, _ADAPTER_FRAMEWORK_MODULES):
                violations.append(f"{module} imports adapter implementation {imported}")
    assert violations == []


def test_workshop_mutation_routes_must_name_a_registered_operation() -> None:
    path = SOURCE_ROOT / "workshop" / "client_api.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    direct_mutation_methods = {"add_post", "add_put", "add_patch", "add_delete"}
    violations: list[str] = []
    registered_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in direct_mutation_methods:
            violations.append(f"line {node.lineno} uses router.{node.func.attr} without an operation binding")
        if isinstance(node.func, ast.Name) and node.func.id == "_register_workshop_capability_route":
            registered_calls += 1
            keyword_names = {keyword.arg for keyword in node.keywords}
            if {"operation_id", "entrypoint"} - keyword_names:
                violations.append(f"line {node.lineno} has an incomplete operation binding")
    assert registered_calls >= 40
    assert violations == []


def test_every_implemented_workshop_capability_has_a_checked_route_binding() -> None:
    path = SOURCE_ROOT / "workshop" / "client_api.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    bound_operations: set[str] = set()
    malformed: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_register_workshop_capability_route":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        operation = keywords.get("operation_id")
        if not isinstance(operation, ast.Constant) or not isinstance(operation.value, str):
            malformed.append(f"line {node.lineno} has a non-literal primary operation binding")
            continue
        bound_operations.add(operation.value)
        additional = keywords.get("additional_operation_bindings")
        if additional is None:
            continue
        if not isinstance(additional, (ast.Tuple, ast.List)):
            malformed.append(f"line {node.lineno} has malformed additional operation bindings")
            continue
        for item in additional.elts:
            if (
                not isinstance(item, (ast.Tuple, ast.List))
                or len(item.elts) != 2
                or not isinstance(item.elts[0], ast.Constant)
                or not isinstance(item.elts[0].value, str)
            ):
                malformed.append(f"line {node.lineno} has malformed additional operation binding")
                continue
            bound_operations.add(item.elts[0].value)

    implemented = {
        capability.operation_id
        for capability in CAPABILITY_REGISTRY
        if capability.presentations[AdapterId.WORKSHOP].implemented
    }
    assert malformed == []
    assert bound_operations == implemented
