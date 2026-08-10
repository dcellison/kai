# Security Remediation Status

This document records the current implementation status of the static security
assessment findings. It is intentionally operational: it distinguishes closed
items from compatibility exceptions so future hardening work does not remove a
working migration path by accident.

Last reviewed: 2026-08-10

## Compatibility rule

Security hardening must preserve existing installed behavior unless the change
includes one of these:

- a compatibility path,
- an explicit migration/check command,
- or an operator-confirmed breaking action.

Do not remove the legacy `WEBHOOK_SECRET` fallback until external GitHub and
generic webhook callers have been verified or migrated.

## Finding status

| # | Finding | Status | Notes |
|---|---|---|---|
| 1 | Global internal API bearer not bound to a user | Closed | Internal API credentials are principal-bound and scope-limited. Request data cannot select a different principal. |
| 2 | Agent API secret reused for GitHub and Telegram webhook signing | Closed with compatibility exception | `KAI_WEBHOOK_SECRET`, `GITHUB_WEBHOOK_SECRET`, `GENERIC_WEBHOOK_SECRET`, and `TELEGRAM_WEBHOOK_SECRET` are separate domains. Persistent agents receive only the principal-bound internal API credential. `WEBHOOK_SECRET` remains as a deprecated external-only fallback for upgraded installs. |
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
- `GITHUB_WEBHOOK_SECRET`: preferred GitHub webhook HMAC secret.
- `GENERIC_WEBHOOK_SECRET`: preferred generic webhook header secret.
- `TELEGRAM_WEBHOOK_SECRET`: Telegram update secret token.
- `WEBHOOK_SECRET`: deprecated compatibility fallback for `/webhook/github` and
  `/webhook` only.

`make config` preserves an existing `WEBHOOK_SECRET` by default. The operator
must explicitly answer `false` to **Retain deprecated WEBHOOK_SECRET fallback**
to remove it from generated configuration.

Before removing fallback support from runtime code:

1. Verify GitHub webhook configuration uses `GITHUB_WEBHOOK_SECRET`.
2. Verify any generic webhook caller uses `GENERIC_WEBHOOK_SECRET`.
3. Run `make config` and explicitly retire `WEBHOOK_SECRET`.
4. Review `make DRY_RUN=1 install`.
5. Deploy with `make install`.
6. Confirm GitHub and generic webhook delivery after deployment.

Only after that migration is confirmed should a later PR remove runtime fallback
acceptance.

## Next safe work

The original assessment findings are remediated or explicitly documented as a
compatibility exception. The next security work should be selected from:

- adding an operator command or diagnostic that reports whether the legacy
  `WEBHOOK_SECRET` fallback is still configured,
- deeper review of the new backend registry and OS-user isolation paths,
- or normal hardening/refactoring of large trust-boundary modules after the
  current behavior is pinned by tests.
