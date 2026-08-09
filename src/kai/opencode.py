"""
OpenCode ACP (Agent Client Protocol) subprocess backend.

Thin adapter over the shared `AcpBackend` layer for OpenCode's specific
argv (`opencode acp`), env-driven model selection
(`OPENCODE_CONFIG_CONTENT`), integer protocol version, and
server-initiated tool-permission request handling. All transport,
lifecycle, context injection, and timeout behavior live in
`kai.acp.AcpBackend`; this module owns only the OpenCode-specific
hooks the base class delegates to.

The OpenCode ACP protocol matches the shared ACP shape for everything
except two harness-specific points:

1. `initialize` requires `protocolVersion` as an INTEGER (1). Goose
   accepts the string "v1"; OpenCode rejects it with a Zod schema
   error and the handshake fails before session/new ever runs.
2. Tool calls in OpenCode emit a server-initiated JSON-RPC request
   `session/request_permission` and BLOCK until the client returns a
   matching-id response with one of the `options` (allow_once /
   allow_always / reject_once). Goose has no equivalent: its
   `--with-builtin developer` extension auto-approves silently.

Model selection flows through `OPENCODE_CONFIG_CONTENT`. The CLI does
not accept `--model` on argv; the only argv flags are
`--cwd`, `--port`, `--hostname`, `--log-level`, `--print-logs`,
`--pure`, `--mdns`, `--mdns-domain`, `--cors`, `--help`, `--version`.
OpenCode reads inline config from the env var at startup; Kai writes
the active model into a one-shot JSON blob per process.

Operator authentication is handled OUTSIDE Kai: `opencode auth login`
writes credentials to `~/.local/share/opencode/auth.json`. The Kai
installer prints a reminder; the wizard does not attempt to manage
OpenCode auth state. On installs with per-user OS isolation
(users.yaml `os_user`), the login must be run AS each target user so
the auth file lands under that user's home; the `sudo -H` wrap the
shared AcpBackend applies points the subprocess at exactly that file.
"""

import json
import logging
import os
from typing import Literal

from kai.acp import AcpBackend
from kai.backend_registry import BackendRegistryError, backend_registry_is_authoritative, resolve_backend_command

log = logging.getLogger(__name__)


# Policy parameter accepted by `handle_opencode_permission_request`.
# `allow_always` matches the conversational backend's auto-approve
# posture (tools must run in chat mode). `reject_once` is the one-shot
# reasoner's posture: memory extraction, PR review, and triage prompts
# must NOT execute tools, so any tool-permission prompt is denied for
# this one call without poisoning the user's broader permission state.
OpenCodePermissionPolicy = Literal["allow_always", "reject_once"]


# Priority list for each policy. The free function below walks the
# server-supplied `options` list using these kinds in order, falling
# back to the first option only if no kind matches. The fallback
# branch carries a distinct rationale tag so a future ACP protocol
# drift surfaces in INFO logs without crashing the read loop.
_OPENCODE_POLICY_PRIORITY: dict[OpenCodePermissionPolicy, tuple[str, ...]] = {
    "allow_always": ("allow_always", "allow_once"),
    "reject_once": ("reject_once", "reject_always"),
}


# ── Shared free functions (used by conversational + one-shot) ────


def concat_opencode_text(prev: str, new: str) -> str:
    """
    Concatenate two OpenCode text chunks, injecting a single space
    when the join would glue a sentence end to a new sentence start.

    OpenCode's `session/update` `agent_message_chunk` notifications
    carry text verbatim from the model. The model usually emits the
    space between sentences inside one chunk or the next, in which
    case verbatim concatenation produces the right output. Sometimes
    it does not: chunk A ends with `"diff."`, chunk B starts with
    `"Now let me read the test files"`, and verbatim concat produces
    `"diff.Now let me read the test files"` (operator-observed
    example). The output reads as a typo and degrades the chat feel
    without changing what the model actually said.

    Heuristic: inject a single space iff `prev` ends in
    sentence-terminating punctuation (`.`, `!`, `?`) AND `new` begins
    with an uppercase letter. The narrow trigger preserves verbatim
    joins for code (`function(`), numbers (`1.2.3`, semantic version
    strings), URLs, markdown headers (`##`), quoted strings, and
    word continuation (`hel` + `lo` -> `hello`). Lowercase sentence
    starts are a known miss; the false-positive cost of widening the
    trigger to cover them is higher than the rare miss.

    Empty inputs round-trip unchanged so the first chunk's
    accumulator initialization (`""` + first chunk) does not inject
    a spurious space.
    """
    if not prev or not new:
        return prev + new
    if prev[-1] in ".!?" and new[0].isupper():
        return prev + " " + new
    return prev + new


def extract_opencode_text_delta(msg: dict) -> str | None:
    """
    Return user-visible assistant text from an OpenCode
    `session/update` notification, or None for non-text shapes.

    OpenCode's `session/update` shape matches Goose's exactly: a
    `params.update.sessionUpdate` discriminator with text under
    `params.update.content.text`. Other discriminators observed
    (`agent_thought_chunk`, `available_commands_update`,
    `usage_update`, `tool_call`, `tool_call_update`) are filtered
    out (return None) so only user-visible assistant text streams
    to the caller.

    Shared between `OpenCodeBackend.extract_text_delta` (the
    conversational backend's hook) and the one-shot reasoner's
    response accumulator so both surface the same set of session
    update shapes as text and skip the same set as non-text.
    Changing the filter here updates both call sites in lockstep.
    """
    if msg.get("method") != "session/update":
        return None
    update = msg.get("params", {}).get("update", {})
    if update.get("sessionUpdate") != "agent_message_chunk":
        return None
    text = update.get("content", {}).get("text", "")
    return text or None


def handle_opencode_permission_request(
    msg: dict,
    *,
    policy: OpenCodePermissionPolicy,
) -> dict | None:
    """
    Build the ACP v1 `RequestPermissionResult` body for an OpenCode
    `session/request_permission` request under the given policy.

    `policy="allow_always"` mirrors the conversational backend's
    posture (tools must execute so the chat agent can do work).
    `policy="reject_once"` is the one-shot reasoner's posture
    (memory extraction / review / triage prompts must not execute
    tools, but the rejection is scoped to this single one-shot
    invocation rather than persisted to user config).

    The selection order per policy is encoded in
    `_OPENCODE_POLICY_PRIORITY`: try each kind in turn, then fall
    back to the first available option (with a distinct rationale
    tag) so a future ACP protocol drift remains observable in INFO
    logs rather than crashing the read loop. Returns None when the
    request shape carries no usable option; the caller logs and
    skips so OpenCode times out the request gracefully.

    Response shape follows the ACP v1 protocol's
    `RequestPermissionResult`: the result body carries an `outcome`
    object with a discriminator (`selected` or `cancelled`) and an
    `optionId` field when `selected`. The shared
    `_send_server_response` wraps this dict under the JSON-RPC
    `result` key, producing `result.outcome.optionId` on the wire.
    Sending a bare `result.optionId` (the pre-#574 shape) is
    structurally invalid against the ACP schema and stalls or kills
    the session. See https://agentclientprotocol.com/protocol/v1/tool-calls.
    """
    if msg.get("method") != "session/request_permission":
        return None
    options = msg.get("params", {}).get("options", []) or []

    # Walk the policy's priority list in order. The first matching
    # `kind` wins; the rationale tag carries the kind name so INFO
    # logs name the exact selection that fired.
    choice: dict | None = None
    rationale: str = "no_usable_option"
    for kind in _OPENCODE_POLICY_PRIORITY[policy]:
        candidate = _pick_option(options, kind)
        if candidate is not None:
            choice = candidate
            rationale = kind
            break
    if choice is None and options:
        # Fallback: take the first option regardless of kind, but tag
        # the rationale so a future protocol drift (a new kind name
        # the priority list does not know about) is observable rather
        # than silent. The fallback fires for both policies because
        # surfacing "the schema changed" is more valuable than
        # silently dropping the permission decision.
        choice = options[0]
        rationale = "fallback_first_option"

    if not choice or "optionId" not in choice:
        log.warning(
            "OpenCode session/request_permission missing a usable option; policy=%s rationale=%s params=%s",
            policy,
            rationale,
            msg.get("params"),
        )
        return None

    option_id = choice["optionId"]
    log.info(
        "OpenCode session/request_permission handled: policy=%s optionId=%s rationale=%s",
        policy,
        option_id,
        rationale,
    )
    return {
        "outcome": {
            "outcome": "selected",
            "optionId": option_id,
        }
    }


def _pick_option(options: list[dict], kind: str) -> dict | None:
    """Return the first option whose `kind` matches, or None."""
    for opt in options:
        if opt.get("kind") == kind:
            return opt
    return None


# ── OpenCode ACP backend ──────────────────────────────────────────


class OpenCodeBackend(AcpBackend):
    """
    OpenCode adapter over the shared AcpBackend layer.

    Provides OpenCode-specific hooks: integer protocol version on the
    initialize handshake, `OPENCODE_CONFIG_CONTENT` env-driven model
    selection, agent_message_chunk parsing (identical shape to Goose),
    and auto-approval of server-initiated
    `session/request_permission` requests so tool calls do not hang.
    """

    # Machine identifier consumed by pool/bot dispatch and tagged
    # into the `memory.recall` line so log analysts can attribute
    # recall behavior to the caller backend.
    backend_name = "opencode"
    # Human-readable label used in error messages and log lines.
    backend_label = "OpenCode"

    # ── Hook implementations ──────────────────────────────────────────

    def build_argv(self) -> list[str]:
        """
        Argv for `opencode acp`.

        The CLI does not accept `--model` on argv; model selection
        flows through `OPENCODE_CONFIG_CONTENT` in `build_env`. `--cwd`
        is supplied via the subprocess `cwd=` argument in
        `AcpBackend._ensure_started`, not as an explicit argv entry.

        OPENCODE_BIN pins an absolute binary path (mirrors codex's
        CODEX_BIN and goose's GOOSE_BIN): when opencode is not on the
        service user's PATH, or the spawn is sudo-wrapped for a
        per-user os_user, the bare name either fails to resolve or
        resolves to a path different from the one the sudoers rule
        pins, and the spawn dies. Falls back to bare "opencode" so
        installs with opencode on PATH keep working without the
        override.
        """
        if backend_registry_is_authoritative():
            try:
                opencode_bin = resolve_backend_command("opencode", allow_bare_fallback=True)
            except BackendRegistryError as e:
                raise RuntimeError(f"OpenCode startup failed: {e}") from e
        else:
            opencode_bin = os.environ.get("OPENCODE_BIN") or "opencode"
        return [opencode_bin, "acp"]

    def build_env(self, base_env: dict[str, str]) -> dict[str, str]:
        """
        Inject the active model into `OPENCODE_CONFIG_CONTENT`.

        OpenCode reads inline JSON config from this env var at startup
        (verified empirically: setting `{"model": "<provider/model>"}`
        changes the model reported in `session/new`'s
        `_meta.opencode.modelId` from the default to the requested ID).
        Model strings use OpenCode's full `provider_id/model_id`
        format (e.g. `anthropic/claude-sonnet-4-6`); see the wizard
        prompt and the README for the supported pattern.

        Only emitted when `self.model` is truthy. Empty model lets
        OpenCode fall back to whatever its own config files specify,
        which is the right behavior for an unconfigured-on-purpose
        adapter (e.g. operator running tests).
        """
        if self.model:
            base_env["OPENCODE_CONFIG_CONTENT"] = json.dumps({"model": self.model})
        return base_env

    def preserved_env_vars(self) -> tuple[str, ...]:
        """
        Env vars the cross-user sudo wrap forwards through env_reset.

        OpenCode reads its model selection from OPENCODE_CONFIG_CONTENT
        (set in build_env). claude and codex carry model selection
        outside the environment (argv flag and per-turn protocol
        params respectively), but for an env-driven backend sudo's
        env_reset strips the model selection itself, not just auth,
        and the wrapped opencode falls back to whatever its per-user
        config files say.
        The five provider key vars cover env-key auth flows centrally;
        the primary auth store (`~/.local/share/opencode/auth.json`,
        written by `opencode auth login` run as the target user) rides
        the HOME rewrite from `sudo -H` instead and needs no
        preservation. No endpoint-override vars appear here because
        opencode has none: custom base URLs are config-file state
        (`provider.<id>.options.baseURL` in opencode.json), which
        reaches the agent through the per-user config under the
        rewritten HOME or through OPENCODE_CONFIG_CONTENT, both
        already covered. KAI_WEBHOOK_SECRET and TMPDIR mirror the
        AcpBackend default (webhook callback auth and the per-os-user
        temp anchor).
        """
        return (
            "OPENCODE_CONFIG_CONTENT",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
            "KAI_WEBHOOK_SECRET",
            "TMPDIR",
        )

    def build_initialize_params(self) -> dict:
        """
        OpenCode requires `protocolVersion` as an integer.

        The shared `AcpBackend.build_initialize_params` default sends
        the string `"v1"` (Goose-compatible). OpenCode's Zod schema
        rejects strings with `"expected number, received string"`,
        breaking the handshake before session/new. Sending integer `1`
        is accepted; OpenCode's response carries `protocolVersion: 1`.
        """
        from kai import __version__

        return {
            "protocolVersion": 1,
            "clientInfo": {"name": "kai", "version": __version__},
        }

    def build_session_new_params(self) -> dict:
        """
        OpenCode's session/new params: cwd + empty mcpServers array.

        Identical to Goose's payload; OpenCode accepts both fields
        without complaint and returns a `sessionId` in the result
        (matching the AcpBackend default `extract_session_id`).
        """
        return {
            "cwd": str(self.workspace),
            "mcpServers": [],
        }

    def extract_text_delta(self, msg: dict) -> str | None:
        """
        Conversational-backend wrapper over
        `extract_opencode_text_delta`. The one-shot reasoner in
        `kai.oneshot` calls the same free function directly so the
        text-discriminator filter stays consistent across both
        callers.
        """
        return extract_opencode_text_delta(msg)

    def combine_text_chunks(self, prev: str, new: str) -> str:
        """
        Smart sentence-boundary concatenation for streamed OpenCode
        text. Overrides the AcpBackend default (verbatim `prev + new`)
        to inject a single space when the chunk boundary glues a
        sentence end to a new sentence start. See
        `concat_opencode_text` for the heuristic. The one-shot
        reasoner calls the free function directly; only the
        conversational backend goes through this hook.
        """
        return concat_opencode_text(prev, new)

    def handle_server_request(self, msg: dict) -> dict | None:
        """
        Conversational-backend hook for OpenCode
        `session/request_permission` requests. Delegates to
        `handle_opencode_permission_request` with policy
        `"allow_always"` so tool calls auto-approve in chat. The
        one-shot reasoner uses policy `"reject_once"` against the
        same free function so memory extraction / review / triage
        prompts cannot execute tools.
        """
        return handle_opencode_permission_request(msg, policy="allow_always")
