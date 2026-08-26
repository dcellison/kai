# Telegram adapter boundary

## Completion rule

Kai core and Workshop must install, import, start, execute agents, commit
post-run effects, remember, schedule, publish proactive output, route
integrations, and settle deliveries without the Telegram extra installed or a
Telegram adapter enabled. Adding a transport may contribute only identity and
channel bindings, ingress translation, capabilities, presentation, and a
delivery worker. Composition and configuration must register that adapter, but
core feature services must not change for the new transport.

CI enforces this rule in two dependency environments:

- `core-workshop` installs the base package without `python-telegram-bot`,
  rejects any attempted Telegram SDK import, and exercises the core lifecycle,
  fresh provisioning, command execution, post-run memory, scheduling,
  integration routing, proactive publication, and transport-neutral delivery;
- `telegram-adapter` installs the `telegram` extra and exercises Telegram
  lifecycle and delivery behavior separately.

The static architecture test additionally rejects Telegram SDK imports outside
the adapter implementation, imports of Telegram implementation modules from
core or feature services, Telegram-named identity or cancellation APIs in core
services, Telegram state in the shared HTTP host, and adapter presentation
types re-exported from the Workshop package. `kai.main` is the sole composition
root and loads the adapter dynamically only when it is enabled. Dormant adapter
configuration is not parsed and cannot veto Workshop-only startup.

## Remaining Telegram references

The following references are intentional and do not grant Telegram authority
over core behavior.

| Location or category | Reason retained | Boundary |
|---|---|---|
| `telegram_adapter`, `bot`, `telegram_context`, `telegram_http`, `memory_command`, `telegram_utils` | Telegram lifecycle, update authentication, command and media presentation, ingress, formatting, and SDK calls | Optional adapter implementation; statically forbidden as a core dependency |
| `workshop/telegram_delivery*` | Convert canonical outbox work into Telegram API calls and record adapter outcomes | Optional delivery adapter/worker; core owns the outbox and settlement contract |
| `telegram_contract` | SDK-free capability declaration used during composition and contract testing | Metadata only; importing it cannot import or contact Telegram |
| `workshop/streaming_preview` and corresponding session facade | Persist and validate Telegram preview/edit state used by the adapter | Presentation state only; imported explicitly by adapter paths and not re-exported from the Workshop package |
| Telegram update queue and preview tables in `sessions`/schema | Durable adapter ingress and retained adapter presentation state | Used only through Telegram adapter paths; not canonical identity or execution keys |
| Telegram identities and channel bindings in Workshop schema, bootstrap, linking, diagnostics, and operator commands | Map one optional external transport to canonical principals and channels; preserve installed migrations | Generic core resolvers consume provider/transport values; bindings are data, not owner/runtime keys, and disabling the adapter makes them ineligible for delivery |
| installer/configuration/status references | Install the optional extra, token, webhook mode, allowlist, and adapter policy | Deployment composition and truthful diagnostics only |
| historical JSONL, numeric-directory, migration, and archive references | Preserve owner-gated historical evidence and migration receipts | Explicitly non-authoritative and never a fallback for protected execution |
| Telegram-focused tests and documentation | Verify the optional adapter and record prior migration decisions | Separate from the Telegram-free core gate |

Any new Telegram SDK import, core import of a Telegram implementation module,
caller-supplied transport identity in a core API, direct Telegram send from a
feature service, or Telegram-keyed runtime decision is a regression.

## Qualification matrix

Automated gates cover a base Workshop-only environment, the optional Telegram
extra, a retained-but-disabled Telegram binding, independent multi-adapter
fan-out outcomes, and a non-Telegram test adapter spanning ordinary replies,
scheduling, proactive publication, integration routing, and post-run effects.

Installed completion evidence is recorded against the backends actually
configured on the qualification host. It must include Workshop-only and hybrid
operation, restart continuity, canonical memory and scheduling, optional
Telegram delivery when enabled, and healthy authoritative status. Supported
but unconfigured backends do not block the adapter-boundary result; their
backend-specific harness tests remain a separate compatibility obligation.
