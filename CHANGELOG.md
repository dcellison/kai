# Changelog

This file records the notable operator-facing changes in each Kai release.

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
