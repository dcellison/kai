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
OpenCode auth state.
"""

import json
import logging

from kai.acp import AcpBackend

log = logging.getLogger(__name__)


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

    # Machine identifier consumed by pool/bot dispatch and by the
    # shadow-mode recall logger to tag `memory.recall_shadow` lines
    # with the caller backend.
    backend_name = "opencode"
    # Human-readable label used in error messages and log lines.
    backend_label = "OpenCode"

    # ── Hook implementations ──────────────────────────────────────────

    def build_argv(self) -> list[str]:
        """
        Static argv for `opencode acp`.

        The CLI does not accept `--model` on argv; model selection
        flows through `OPENCODE_CONFIG_CONTENT` in `build_env`. `--cwd`
        is supplied via the subprocess `cwd=` argument in
        `AcpBackend._ensure_started`, not as an explicit argv entry.
        """
        return ["opencode", "acp"]

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
        Return text from an OpenCode `agent_message_chunk` notification.

        OpenCode's `session/update` shape matches Goose's exactly: a
        `params.update.sessionUpdate` discriminator with text under
        `params.update.content.text`. Other discriminators observed
        (`agent_thought_chunk`, `available_commands_update`,
        `usage_update`, `tool_call`, `tool_call_update`) are filtered
        out (return None) so only user-visible assistant text streams
        to the Telegram client.
        """
        if msg.get("method") != "session/update":
            return None
        update = msg.get("params", {}).get("update", {})
        if update.get("sessionUpdate") != "agent_message_chunk":
            return None
        text = update.get("content", {}).get("text", "")
        return text or None

    def handle_server_request(self, msg: dict) -> dict | None:
        """
        Auto-approve OpenCode `session/request_permission` requests.

        OpenCode emits a server-initiated JSON-RPC request when a tool
        needs a permission decision; the prompt does not progress
        until the client returns a matching-id response. The request's
        `options` array always carries an entry with
        `kind: "allow_always"`; returning its `optionId` matches the
        Goose `--with-builtin developer` posture (auto-approve so
        agent tasks can actually execute tools), without baking the
        decision into OpenCode's persistent config.

        Defensive fallback: if the request shape changes and we cannot
        find an allow_always option, look for any allow-shaped option,
        then for any optionId at all, and only then bail with None
        (which logs and skips at the shared-layer call site, letting
        OpenCode time out the request gracefully instead of crashing
        the read loop).
        """
        if msg.get("method") != "session/request_permission":
            return None
        options = msg.get("params", {}).get("options", []) or []
        choice = (
            _pick_option(options, "allow_always")
            or _pick_option(options, "allow_once")
            or (options[0] if options else None)
        )
        if not choice or "optionId" not in choice:
            log.warning(
                "OpenCode session/request_permission missing a usable option; params=%s",
                msg.get("params"),
            )
            return None
        return {"optionId": choice["optionId"]}


def _pick_option(options: list[dict], kind: str) -> dict | None:
    """Return the first option whose `kind` matches, or None."""
    for opt in options:
        if opt.get("kind") == kind:
            return opt
    return None
