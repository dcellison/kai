# Changelog

This file records the notable operator-facing changes in each Kai release.

## [2.1.0] - 2026-08-26

Kai 2.1 introduces Kai Workshop as a first-class browser client and establishes
the qualified boundary between Kai's canonical core and its optional client
adapters. Telegram remains fully supported, but no longer supplies Kai's
application lifecycle, identity, runtime, transcript, scheduling, memory,
integration, or publication authority.

### Highlights

- A LAN-accessible React Workshop client with secure enrollment, revocable
  sessions, canonical conversation history, live updates, Markdown rendering,
  resilient scrolling, run activity, cancellation, and a detailed agent trace
  inspector.
- Durable canonical principals, channels, agents, runtime profiles, messages,
  runs, attempts, artifacts, deliveries, schedules, and post-run effects.
- Browser-native uploads and authorized previews for images, documents, and
  audio, plus transport-neutral artifact access for agents.
- Canonical settings and workspace controls, notification feeds, GitHub review
  and issue-triage automation, generic-webhook routing, and a read-only semantic
  memory explorer.
- Workshop-only, Telegram-only, and hybrid application composition. Telegram
  is now installed through the optional `telegram` dependency extra and is
  supervised as an adapter rather than as the application host.

### Canonical runtime and state

- Root-owned runtime profiles select the backend, provider, OS execution user,
  model and timeout policy, service grants, and workspace policy independently
  of Telegram configuration.
- Backend processes and persistent execution state are addressed by canonical
  runtime-profile and conversation identities. Compatibility execution reads,
  rollback dual writes, and Telegram-shaped runtime keys are retired.
- Canonical messages are the live transcript and cold-start context authority.
  Legacy JSONL records remain classified archives and are not read or written
  by protected execution.
- The core-owned scheduler creates canonical work and survives restart without
  depending on Telegram's job queue.
- Internal agent APIs resolve a scoped canonical execution context from their
  credentials and reject caller-supplied identity selectors.
- GitHub automation, generic integrations, proactive messages, and ordinary
  replies publish canonically and reach enabled transports through durable,
  adapter-pluggable delivery plans and outbox workers.
- Common post-run effects, including semantic-memory ingestion, are owned by a
  durable canonical worker rather than by a client handler.

### Workshop client

- Browser enrollment uses one-time, hashed enrollment credentials and produces
  revocable device sessions without requiring a Telegram identity.
- Conversation submission, streaming progress, finalization, cancellation,
  restart recovery, and provider-session continuity use the same durable run
  lifecycle across clients.
- The conversation pane supports recent-history windows, jump-to-latest,
  Markdown, uploads, previews, direct-message identification, collapsible and
  resizable navigation, and an Atom One Dark-inspired interface.
- The run inspector exposes backend-neutral tool, command, file, diff, and
  progress traces without granting the browser direct backend authority.
- The memory explorer provides principal-scoped statistics, filtering, search,
  provenance, and source inspection. Editing and management remain planned
  follow-up work rather than part of this release.

### Telegram adapter boundary

- Telegram private text and media enter the canonical command, artifact, run,
  transcript, post-run, and delivery boundaries used by other clients.
- Telegram-specific update parsing, authentication, external identity and
  binding, Markdown, keyboards, media download, streaming previews, Bot API
  delivery, retries, webhook/polling lifecycle, and diagnostics remain adapter
  responsibilities.
- Disabling Telegram prevents dormant bindings from planning Telegram
  deliveries and does not require dormant Telegram human configuration to be
  parsed.
- Core and Workshop CI runs without the Telegram extra and blocks Telegram SDK
  imports. A separate adapter job installs and verifies Telegram behavior.
- Retained Telegram bindings, delivery records, migrations, and archives are
  compatibility or audit data, not core runtime authority.

These boundaries are complete against the automated architecture gates and
installed qualifications recorded under epic #917. They are not a claim that
future adapter development can never uncover another coupling; any such
finding should be treated as a boundary regression.

### Memory, security, and operations

- Semantic memory now carries canonical principal, channel, agent,
  runtime-profile, project, and source provenance where applicable.
- Legacy-default memory scopes fail closed, recalled memory is explicitly
  treated as untrusted data, vector-store and telemetry permissions are
  private, and nightly snapshots cover the semantic-memory corpus.
- Memory embeddings load from the local cache without startup requests to
  Hugging Face, and extraction plus vector-store work drains cleanly during
  shutdown.
- Per-principal uploads, history, managed homes, preferences, and memory use
  canonical namespaces with protected ownership and modes.
- Launchd generations are serialized, Workshop streams and semantic memory
  close during graceful shutdown, and installation reports its readiness wait
  instead of appearing to stall.
- Dependency constraints are audited and enforced in CI.

### Upgrade from 2.0.0

For an existing protected installation:

1. Update the working tree to the 2.1.0 release.
2. Optionally preview the installation with `make DRY_RUN=1 install`.
3. Run `make install`. The installer performs idempotent canonical-state and
   storage migrations, preserves legacy data as non-authoritative archives,
   and waits for full application readiness.
4. Run `make install-status`. Confirm that the canonical runtime, transcript,
   memory, operational, delivery, internal-API, post-run, and transition
   diagnostics are active or clean as applicable.
5. Verify a Workshop conversation and every enabled client adapter. For a
   hybrid installation, verify both Workshop and Telegram.

Re-run `make config` only when changing client-adapter mode, listener settings,
secrets, or other operator configuration. Existing configured backend accounts
remain sufficient; Kai does not require accounts for unused backends.

### Qualification and known boundaries

- Automated common-contract coverage exercises Claude, Codex, Goose, OpenCode,
  and Pi without invoking provider models. Installed live qualification covers
  the runtime profiles that are configured and authenticated on the host.
- Workshop-only and restored-hybrid installed qualifications passed, including
  execution, cancellation, restart recovery, canonical history, scoped memory,
  artifacts, scheduling, integrations, and Telegram delivery.
- The final corrective boundary change passed separate core/Workshop and
  Telegram-adapter CI jobs, the full Python suite, lint, formatting, strict
  typing, and dependency audit.
- Unreviewed legacy-default semantic memories remain quarantined and
  fail-closed until explicitly classified. This does not weaken canonical
  memory authority.
- Historical numeric directories and JSONL data may remain as documented
  archives. Protected runtime reads and writes do not use them.
- The local-process protected runtime remains a trusted-host compatibility
  mode. Isolated workers remain the intended stronger multi-user boundary.

**Full comparison:** [v2.0.0...v2.1.0]

## [2.0.0] - 2026-08-11

Kai 2.0 is the stable Telegram-first, multi-backend release. It replaces the
single-harness assumptions in Kai 1.x with explicit backend selection, adds
project-scoped semantic memory, and closes the findings from the 2026 static
security assessment.

### Highlights

- Five first-class agent backends: Claude Code, OpenAI Codex CLI, Goose,
  OpenCode, and Pi.
- Explicit installation-wide backend selection with optional per-user and
  per-role model overrides. No backend has implicit priority.
- An administrator-owned backend registry for protected installs. Runtime
  configuration selects registered backend identifiers and models rather than
  user-supplied executable paths.
- Semantic memory with extracted facts and episodes, speaker attribution,
  project scoping, provenance, migration and reclassification tools, and a
  Telegram memory browser.
- Durable Telegram webhook ingestion, compensated schedule updates, graceful
  background-task draining, and nonzero exit status after unexpected crashes.
- `protected` and `single_user` deployment modes, a real non-mutating install
  preview, recoverable virtual-environment updates, and an authoritative
  `make install-status` diagnostic.

### Security

- Internal API credentials are bound to authenticated principals and explicit
  scopes. Notification credentials cannot exercise user or agent authority.
- GitHub operations require an authorized repository and, in protected mode,
  the initiating user's stored GitHub identity. Notification subscriptions do
  not expand operation authority.
- External-service proxy access is authorized per user, and service keys stay
  in the outer Kai process.
- GitHub, generic, Telegram, and internal API credentials use separate secret
  domains. The former `WEBHOOK_SECRET` fallback is unsupported and ignored.
- Protected installs require distinct target OS users for interactive agents,
  validate protected configuration ownership and modes, and constrain backend
  execution through generated exact-command sudoers rules.
- SQLite state, memory, preferences, home directories, uploaded files, file
  sending, and backend image handoff use private per-user boundaries.
- TOTP state fails closed when enabled and gates sensitive command paths. A
  deployment with TOTP disabled remains supported.
- Issue-triage mutations and service-proxy URL construction are constrained by
  administrator policy.

The detailed finding-by-finding record is in
[Security Remediation Status](SECURITY_REMEDIATION_STATUS.md).

### Upgrade from 1.4.0

Kai 2.0 contains intentional configuration and security-boundary changes. For
an existing protected installation:

1. Install and authenticate every backend that will be selected, using each
   target user's own OS account where applicable.
2. Run `make config`. Keep the existing Telegram bot token, choose the desired
   default backend, and review the generated configuration. `users.yaml` is
   mandatory; per-user backend entries may override the installation default.
3. Ensure external webhook callers use their dedicated credentials:
   `GITHUB_WEBHOOK_SECRET` for GitHub and `GENERIC_WEBHOOK_SECRET` for the
   generic endpoint. `TELEGRAM_WEBHOOK_SECRET` remains specific to Telegram
   webhook delivery.
4. Optionally preview with `make DRY_RUN=1 install`, then run `make install`.
   The Make target invokes `sudo` internally.
5. Run `make install-status`, then verify Telegram and any configured webhook
   deliveries.

`install.conf` is the artifact produced by `make config` and may contain
secrets. After a successful protected install, deployed secrets live in
root-owned `/etc/kai/env`, so the artifact may be deleted and regenerated when
configuration changes are needed.

The old `AGENT_BACKEND` name is migrated to `DEFAULT_BACKEND` during upgrade.
Model defaults now come from the backend/provider/role registry; explicit
per-user model baselines belong in the `models:` map in `users.yaml`.

### Backend authentication

- Claude Code and Codex use their respective local subscription or CLI
  authentication for the target OS user.
- Goose uses its configured provider and that provider's authentication.
- OpenCode uses OpenCode's own provider/model catalog and authentication.
- Pi uses Pi's provider/model selection and the target OS user's Pi
  authentication.

Kai requires only the backends actually selected by the installation or a
user; it does not require all five to be installed.

### Qualification and compatibility boundary

The release commit passed install-constraint validation, lint and formatting,
the maintained Pyright baseline, the full automated test suite, and the Python
dependency audit. A protected macOS installation was upgraded in place and
verified at version 2.0.0, including Telegram conversation and GitHub webhook
notification delivery. All five backend harnesses were exercised during the
2.0 development cycle.

The protected host-process runtime remains a trusted-host compatibility mode.
Kai warns when a registered executable can be modified by an agent OS user;
host executables are not intended to provide hostile multi-user isolation.
Worker isolation and executable provenance belong to the future Kai Workshop
runtime boundary.

The `single_user` path remains supported and covered by automated tests, but a
separate manual end-to-end single-user smoke test was deferred for this
release.

**Full comparison:** [v1.4.0...v2.0.0]

[2.0.0]: https://github.com/dcellison/kai/releases/tag/v2.0.0
[v1.4.0...v2.0.0]: https://github.com/dcellison/kai/compare/v1.4.0...v2.0.0
[2.1.0]: https://github.com/dcellison/kai/releases/tag/v2.1.0
[v2.0.0...v2.1.0]: https://github.com/dcellison/kai/compare/v2.0.0...v2.1.0
