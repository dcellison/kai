# Security Remediation Status

This document records the current implementation status of the static security
assessment findings and the operational checks that closed them.

Last reviewed: 2026-08-11

## Compatibility rule

Security hardening must preserve existing installed behavior unless the change
includes one of these:

- a compatibility path,
- an explicit migration/check command,
- or an operator-confirmed breaking action.

## Finding status

| # | Finding | Status | Notes |
|---|---|---|---|
| 1 | Global internal API bearer not bound to a user | Closed | Internal API credentials are principal-bound and scope-limited. Request data cannot select a different principal. |
| 2 | Agent API secret reused for GitHub and Telegram webhook signing | Closed | `KAI_WEBHOOK_SECRET`, `GITHUB_WEBHOOK_SECRET`, `GENERIC_WEBHOOK_SECRET`, and `TELEGRAM_WEBHOOK_SECRET` are separate domains. Persistent agents receive only the principal-bound internal API credential. The former `WEBHOOK_SECRET` credential is unsupported and ignored. |
| 3 | GitHub reads and mutations used shared outer-process identity | Closed | Protected installs require per-user GitHub tokens for user-initiated GitHub operations. Repository access is authorized through admin-controlled per-user configuration. |
| 4 | Service UID sessions could inherit daemon authority | Closed | Protected installs require distinct target OS users and use a backend registry rather than user-supplied binary paths. Generated sudoers remains exact-command scoped. |
| 5 | Notification destinations could become inbound Telegram principals | Closed | Inbound authorization principals and outbound notification destinations are separate. GitHub notifications can still target configured Telegram groups. |
| 6 | TOTP fail-open and incomplete sensitive-handler coverage | Closed | TOTP state now fails closed when configured, uses safer attempt handling, and gates sensitive command paths while respecting disabled TOTP. |
| 7 | Filesystem and secret-at-rest isolation gaps | Closed | SQLite and token input are protected; per-user data directories, Codex image staging, send-file handoff, upload handoff, and deferred user file reads now respect per-user confinement. |
| 8 | Issue triage trusted model output as authorization | Closed | Triage label and project mutations are constrained by configured allowlists/existing repository labels. |
| 9 | Service proxy `allow_path_suffix` could change URL origin | Closed | Path suffix handling pins the final URL origin. |
| 10 | Accepted work could be lost or duplicated around failures | Closed | Telegram webhook work is durably queued, background tasks drain on shutdown, schedule registration/update failures are compensated, malformed startup jobs are isolated, and unexpected crashes exit nonzero. |
| 11 | CI and maintainability controls lagged stated standard | Closed for controls; ongoing for refactors | Pyright, dependency constraints, install validation, advisory audit, and module-size reporting are present. Large trust-boundary module decomposition remains normal engineering debt rather than a blocking security fix. |

## Webhook secret migration status

Current secret domains:

- `KAI_WEBHOOK_SECRET`: internal agent API credential, principal-bound and scoped.
- `GITHUB_WEBHOOK_SECRET`: GitHub webhook HMAC secret.
- `GENERIC_WEBHOOK_SECRET`: generic webhook header secret.
- `TELEGRAM_WEBHOOK_SECRET`: Telegram update secret token.
- `WEBHOOK_SECRET`: unsupported; ignored by runtime authentication.

The privileged `make install-status` diagnostic confirmed that the deployed
environment uses both named external webhook secrets and contains no
`WEBHOOK_SECRET`. It reports variable presence only, never values.

Runtime fallback removal is enforced at every configuration boundary:

- runtime routes accept only their named credentials;
- `make config` never carries `WEBHOOK_SECRET` into a regenerated artifact;
- `make install` strips it from older artifacts before writing `/etc/kai/env`,
  or fails before stopping Kai if either named replacement is missing;
- `make install-status` continues to identify stale deployed or artifact state.

## Next safe work

After deploying runtime fallback removal, confirm that GitHub notifications
still reach their configured Telegram destination and exercise any active
generic webhook caller. The original assessment findings are then fully
remediated, and the next security work is Kai Workspace architecture validation.
Large trust-boundary module decomposition remains normal engineering debt rather
than a blocking security fix.

## Transitional risk: local-process backend executables

The backend registry currently prevents operators from supplying arbitrary
executable paths, but the registered executables may still live in locations
owned by an agent target OS user. An agent running as that user could replace a
shared executable and cross an OS-user boundary when Kai later invokes it for a
different user.

A root-owned managed backend package store was considered, but is deliberately
deferred. Implementing dependency-closure discovery, provenance verification,
promotion, health checks, and rollback for host executables would be a large
transitional subsystem. Kai Workspace instead assigns binary/image preparation
to its worker Runtime Backend contract and makes isolated container workers the
personal-production boundary. The current server-process runtime is therefore
classified as a trusted-host compatibility and migration mode.

As an interim control, install reports a non-blocking warning when ordinary
ownership/group/mode checks show that a registered command, its resolved
executable, or either path chain can be modified by the service user or a
configured agent `os_user`. This makes the limitation visible without breaking
working Homebrew and other operator-managed installations. The admin-owned
registry, fixed backend identifiers, exact-command sudoers rules, and existing
group/other-writable executable rejection remain in force.

Initial inspection of the currently installed macOS executables found no
non-system dynamic-library dependencies or package-relative runtime search
paths. That is favorable evidence for relocation, but it is not proof that the
programs have no runtime dependency on surrounding files. Strict macOS code
signature verification also does not currently provide a uniform integrity
mechanism across all four installed backends. These observations are retained
as context if a protected host-process package store is reconsidered, but that
store is not part of the current remediation plan.
