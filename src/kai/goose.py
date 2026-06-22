"""
Goose ACP (Agent Client Protocol) subprocess backend.

Thin adapter over the shared `AcpBackend` layer for Goose's specific
argv (`goose acp --with-builtin developer`), env vars (GOOSE_MODEL,
GOOSE_PROVIDER), session/new params (cwd + empty mcpServers), and
streaming notification shape (agent_message_chunk). All transport,
lifecycle, context injection, and timeout behavior lives in
`kai.acp.AcpBackend`; this module owns only the Goose-specific hooks
the base class delegates to.

The Goose ACP protocol:
    Startup:  initialize -> session/new (handshake)
    Input:    session/prompt (JSON-RPC request with sessionId + prompt)
    Output:   session/update (streaming notifications, no id field)
    Finish:   JSON-RPC result with matching id + stopReason
"""

import logging
import os

from kai.acp import AcpBackend

log = logging.getLogger(__name__)


# Map Kai logical model names to Anthropic model IDs for GOOSE_MODEL.
# Values are the API's undated alias forms, the same IDs the claude
# CLI resolves its short aliases to, so the two backends agree on
# which SKU a logical name means. The aliases pin a family version
# (claude-opus-4-8 does not track a future 4.9), so this map needs a
# bump on each Anthropic release. The .get(key, key) fallback passes
# unrecognized values through unchanged, so full model IDs (e.g.
# "claude-opus-4-7" to pin the previous generation) work without
# being in the map; validate_model_for_backend admits any claude-*
# ID on goose-on-anthropic so the pass-through is reachable from
# /model and per-user config.
_ANTHROPIC_MODEL_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
}


# Map Kai provider keys to goose's wire-level provider names. Kai uses
# "deepseek" as its provider key everywhere (BACKEND_PROVIDERS, the
# model registry, per-user config; shared with the opencode backend),
# but goose ships DeepSeek as a declarative provider named
# "custom_deepseek"; passing the bare name fails with "Unknown
# provider: deepseek" before any API call. The .get(key, key) fallback
# passes every other provider through unchanged, so the map only grows
# when goose names a provider differently than Kai does.
_GOOSE_PROVIDER_MAP: dict[str, str] = {
    "deepseek": "custom_deepseek",
}


def goose_provider_id(provider: str) -> str:
    """
    Translate a Kai provider key to goose's wire-level provider name.

    Used by every site that hands a provider name to the goose binary:
    `GooseBackend.build_env` (GOOSE_PROVIDER) and the `goose run`
    one-shot argv in review and triage (`--provider`). Kai-level
    surfaces (wizard, /settings, users.yaml, the model registry) keep
    the Kai key; only the goose wire name differs.
    """
    return _GOOSE_PROVIDER_MAP.get(provider, provider)


# ── Goose ACP backend ─────────────────────────────────────────────


class GooseBackend(AcpBackend):
    """
    Goose adapter over the shared AcpBackend layer.

    Provides Goose-specific hooks (argv, env vars, session/new params,
    streaming notification parsing). Lifecycle, transport, context
    injection, and idle/response timeout behavior are inherited from
    AcpBackend and shared with any other ACP-based backend.

    All message sends are serialized via the AcpBackend lock to prevent
    interleaving. The developer builtin extension auto-approves all
    tool calls, so no permission write-back is needed.
    """

    # Machine identifier consumed by pool/bot dispatch and tagged
    # into the `memory.recall` line so log analysts can attribute
    # recall behavior to the caller backend.
    backend_name = "goose"
    # Human-readable label used in error messages and log lines.
    backend_label = "Goose"

    # ── Hook implementations ──────────────────────────────────────────

    def build_argv(self) -> list[str]:
        """
        Argv for `goose acp` with the developer builtin enabled.

        Goose injects the model via GOOSE_MODEL env, not argv, so model
        does not appear here.

        GOOSE_BIN pins an absolute binary path (mirrors codex's
        CODEX_BIN): when goose is not on the service user's PATH, or
        the spawn is sudo-wrapped for a per-user os_user, the bare
        name either fails to resolve or resolves to a path different
        from the one the sudoers rule pins, and the spawn dies. Falls
        back to bare "goose" so installs with goose on PATH keep
        working without the override.
        """
        goose_bin = os.environ.get("GOOSE_BIN") or "goose"
        return [goose_bin, "acp", "--with-builtin", "developer"]

    def build_env(self, base_env: dict[str, str]) -> dict[str, str]:
        """
        Add GOOSE_MODEL (and conditionally GOOSE_PROVIDER) to the env.

        Kai's logical model names ("sonnet", "opus", "haiku") only
        apply to the Anthropic provider. Other providers require full
        model IDs set via user config (users.yaml or /settings); those
        pass through unchanged.

        GOOSE_PROVIDER tells the goose binary which provider backend to
        talk to (openai, anthropic, google, etc.). Without it,
        session/new fails with "Internal error" against a binary that
        has no default provider configured. Kai's wizard writes
        DEFAULT_PROVIDER to /etc/kai/env for its own bookkeeping, but the
        goose binary reads the GOOSE_-prefixed name; the translation
        happens here so the two layers stay decoupled. The value runs
        through `goose_provider_id` because goose's wire-level provider
        names can differ from Kai's keys (deepseek is custom_deepseek
        on the goose side).

        Guarded on `self.provider` truthiness because it can be the
        empty-string default for the claude backend pre-wiring path,
        and exporting GOOSE_PROVIDER="" would confuse goose more than
        omitting it.
        """
        if self.provider == "anthropic":
            mapped = _ANTHROPIC_MODEL_MAP.get(self.model, self.model)
        else:
            mapped = self.model
        if mapped:
            base_env["GOOSE_MODEL"] = mapped
        if self.provider:
            base_env["GOOSE_PROVIDER"] = goose_provider_id(self.provider)
        return base_env

    def preserved_env_vars(self) -> tuple[str, ...]:
        """
        Env vars the cross-user sudo wrap forwards through env_reset.

        Goose reads its model and provider selection from GOOSE_MODEL
        / GOOSE_PROVIDER (set in build_env), unlike claude and codex,
        which pass model on argv. Without preservation, sudo's
        env_reset strips the model selection itself, not just auth,
        and the wrapped goose falls back to whatever its per-user
        config files say. The five provider key vars cover env-key
        auth flows centrally; keychain-based auth (per-user
        `goose configure`) rides the HOME rewrite from `sudo -H`
        instead and needs no preservation. The endpoint-override vars
        (ANTHROPIC_HOST, OPENAI_HOST plus OPENAI_BASE_PATH,
        OLLAMA_HOST; verified against goose's provider docs) keep a
        custom-endpoint install routed at its proxy or gateway under
        the wrap, mirroring the claude / codex precedent of
        preserving each backend's base-URL override. DeepSeek and
        google have no host vars and ride their API keys alone.
        KAI_WEBHOOK_SECRET and TMPDIR mirror the AcpBackend default
        (webhook callback auth and the per-os-user temp anchor).
        """
        return (
            "GOOSE_MODEL",
            "GOOSE_PROVIDER",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_HOST",
            "OPENAI_HOST",
            "OPENAI_BASE_PATH",
            "OLLAMA_HOST",
            "KAI_WEBHOOK_SECRET",
            "TMPDIR",
        )

    def build_session_new_params(self) -> dict:
        """
        Goose's session/new params: cwd + empty mcpServers array.

        `mcpServers` must be an array (even if empty) - an object
        causes a deserialization error in Goose.
        """
        return {
            "cwd": str(self.workspace),
            "mcpServers": [],
        }

    def extract_text_delta(self, msg: dict) -> str | None:
        """
        Return text from a Goose `agent_message_chunk` notification.

        Goose's session/update notifications carry an `update` object
        with a `sessionUpdate` discriminator. The `agent_message_chunk`
        shape carries assistant text; agent_thought_chunk, tool_call,
        and tool_call_update are filtered out (return None).
        """
        if msg.get("method") != "session/update":
            return None
        update = msg.get("params", {}).get("update", {})
        if update.get("sessionUpdate") != "agent_message_chunk":
            return None
        text = update.get("content", {}).get("text", "")
        return text or None
