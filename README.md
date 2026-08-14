# Kai

[![CI](https://img.shields.io/github/actions/workflow/status/dcellison/kai/ci.yml?branch=main&label=CI)](https://github.com/dcellison/kai/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue)](https://python.org)
[![License](https://img.shields.io/github/license/dcellison/kai)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/dcellison/kai?label=version)](https://github.com/dcellison/kai/releases)

Kai is a local, multi-client personal engineering system: a persistent AI collaborator with repo-aware coding, memory, scheduling, PR review, and multi-backend operation.

Run Kai on your own machine, reach it through the Workshop browser or Telegram, and give it real access to your local workspaces. Kai can inspect repositories, run shell commands, write code, review pull requests, triage issues, remember durable context, handle files, and run scheduled jobs while staying under your control. Your machine, your data, your rules.

For full setup and operations guides, see the [Kai Wiki](https://github.com/dcellison/kai/wiki).

## Why Kai Exists

Most AI coding tools are either interactive terminals or hosted chat surfaces. Kai is built for a different operating model: a long-running local service available through authenticated client adapters.

- **Local-first authority:** Kai runs on hardware you control and works against your local filesystem, shell, git repos, and tools.
- **Multiple client surfaces:** Workshop provides the native browser experience; Telegram remains an optional mobile client with files, voice notes, commands, and notifications.
- **Persistent agent sessions:** Each user gets a lazily-created subprocess with durable context and idle eviction.
- **Memory across sessions:** Kai preserves identity, personal memory, and conversation history so useful context survives restarts and workspace switches.
- **Background engineering workflows:** PR review, issue triage, webhooks, reminders, and condition-monitoring jobs run outside the active chat session.
- **Multi-backend operation:** Each user can run through any installed supported backend, with an explicit installation default and optional per-user overrides.
- **Multi-user isolation:** One Kai instance can serve multiple Workshop humans and optional Telegram identities with isolated history, files, settings, and OS-level process separation.

## Core Capabilities

| Capability | What Kai Does |
|---|---|
| Repo-aware coding | Runs an agent inside local workspaces with shell, filesystem, git, and web access. |
| Workspaces | Switches between projects by name and keeps per-workspace settings. |
| Memory | Maintains identity, durable user memory, semantic recall, and searchable conversation history. |
| Scheduling | Runs reminders, recurring jobs, and condition monitors from Telegram or HTTP. |
| GitHub automation | Reviews PRs, triages issues, routes notifications, and reacts to webhook events. |
| File exchange | Accepts files from Telegram, exposes their local paths to the agent, and can send files back. |
| Voice | Supports local voice transcription and optional text-to-speech responses. |
| Multi-user operation | Isolates canonical principals, runtime profiles, workspaces, files, history, jobs, settings, and OS accounts. |

## How It Works

```text
Workshop browser --\
                    -> Kai service -> per-user agent backend
Telegram (optional)-/                    -> local workspace, shell, git, files, web, services
```

Kai has two layers. The outer Python service owns client adapters, HTTP, scheduling, authentication, persistence, webhooks, file exchange, and per-user routing. The inner agent backend does the thinking and acting inside a local workspace. Backend subprocesses are created lazily per user and evicted after an idle timeout, so resource use follows active users rather than registered users.

This is not an API relay bot. The inner backend is a full coding-agent runtime with local tools and project context. Kai gives that runtime a durable home, authenticated client surfaces, scheduled execution, event-driven inputs, memory, and a security model designed around the fact that it can take real action.

## Backend Options

In Kai, a backend is more than a model provider. Each backend is a full coding harness with its own protocol, tool behavior, authentication path, context handling, model surface, and failure modes. Kai normalizes lifecycle and routing around those harnesses while preserving the differences that matter.

| Backend | Runtime | Model Selection Shape | Notes |
|---|---|---|---|
| Claude Code | `claude` CLI | Claude aliases and full model IDs | Uses Claude Code's local authentication. |
| OpenAI Codex CLI | `codex` CLI | Codex CLI model IDs | Uses Codex's own model catalog, separate from OpenAI API model lists. |
| Goose | `goose acp` | Provider-native model IDs | ACP backend with provider selected through Goose configuration or env. |
| OpenCode | `opencode acp` | `provider/model` IDs | ACP backend with model resolution owned by OpenCode. |
| Pi | `pi --mode rpc` | `provider/model[:thinking]` IDs | JSONL RPC backend using the target OS user's Pi authentication; bounded one-shot tasks disable tools and project resources. |

Kai does not require every supported backend to exist on every machine. Every backend selected as the installation default or in a user's configuration must be installed and authenticated for the OS account that will run it.

## Quick Start

Requirements:

- Python 3.13+
- A Telegram bot token and Telegram user ID only if enabling the optional Telegram client
- At least one supported agent backend installed and authenticated
- Sudo 1.9.3+ for protected multi-user installs (`CWD`/`-D` support)

Install Kai for local development:

```bash
git clone git@github.com:dcellison/kai.git
cd kai
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make config
```

`make config` runs without `sudo`. It discovers the supported backend CLIs installed on the machine, requires an explicit installation default when more than one is available, and writes `install.conf` as the configuration artifact for the selected deployment and client modes. It does not accept user-supplied backend executable paths. Model defaults come from Kai's backend/provider/role model registry. In protected installs, conversational model baselines belong to `runtime-profiles.yaml`; operator-set PR-review and issue-triage baselines remain in each human's `models:` map in `users.yaml`. Users can change their active conversational model with `/model`.

The client-mode choices are explicit and reversible:

- `hybrid` enables Workshop and Telegram and is the upgrade-safe default;
- `workshop-only` switches an existing protected installation with canonical humans and runtime profiles to the browser client without constructing a Telegram application, contacting Telegram, or retaining a bot token;
- `telegram-only` runs Telegram without publishing Workshop client routes.

Re-run `make config`, select a different client mode, and apply the installation to change modes. Existing configurations without an explicit client-mode setting remain hybrid. `make install-status` reports both the deployed policy and the current `install.conf` artifact without exposing the Telegram token.

The initial Workshop-only qualification is deliberately an installed-mode switch, not a fresh-host bootstrap path. Fresh protected and single-user Workshop-only provisioning still needs a canonical administration flow for creating the first human and runtime assignment; the wizard refuses those incomplete combinations instead of generating a configuration that cannot start.

For a `single_user` deployment, `make config` also writes the runtime files under the operator's account. Start Kai from the checkout:

```bash
make run
```

For a `protected` deployment, preview and apply the staged configuration:

```bash
make DRY_RUN=1 install
make install
make install-status
```

`make install` invokes `sudo` internally and installs source, data, and secrets under separate protected system directories. It also generates the admin-owned `/etc/kai/backends.yaml` registry containing the discovered executable paths, allowed model surfaces, and selected default backend. Runtime configuration names backend identifiers; it cannot redirect a protected backend to an arbitrary executable. After a successful protected install, `install.conf` may be deleted because it can contain secrets; re-run `make config` before a later reconfiguration.

On the first protected install, Kai seeds `/etc/kai/runtime-profiles.yaml`
from the configured humans. After that file exists, it is the independent
authority for each Workshop runtime's backend, provider, model baseline,
timeout, OS execution user, service grants, and workspace policy. Later edits
to duplicated execution fields in `users.yaml` do not replace or veto the
protected profile. `users.yaml` remains the Telegram-human bootstrap and
integration-policy input during the Workshop migration.

Edit an installed runtime profile with `sudoedit`, validate it with
`make install-status`, then run `make install` to reconcile OS-owned storage
and restart Kai. The installer preserves explicit profile values and fails
closed if a configured compatibility identity has lost its profile mapping or
if the protected policy references an unavailable backend or invalid model.

Workshop remains loopback-only by default. To reach its browser client directly
from a trusted private network, enter one private IPv4 interface address at the
optional `Workshop LAN address` prompt. Kai then creates a second listener on
the webhook port containing only the Workshop shell, enrollment, timeline,
event-stream, and authenticated command-submission routes; internal agent APIs
and webhook ingress remain bound to loopback. This direct listener uses HTTP,
so use a trusted LAN or put a trusted TLS terminator in front of it.

Protected mode requires every configured Telegram user to have a unique `os_user` that differs from the Kai service account; this keeps persistent agents from inheriting the daemon's protected-config capabilities. Single-user mode runs the agent as the operator account.

Kai-managed per-user identity has one editable source: `<DATA_DIR>/home/<principal_id>/AGENTS.md`. Codex, Goose, OpenCode, and Pi consume that file directly. Claude Code receives a generated `.claude/CLAUDE.md` adapter containing only `@../AGENTS.md`. On upgrade, `make install` migrates existing customized Claude identity content into `AGENTS.md`; if both files contain different customizations, installation stops without choosing or overwriting either one. Instruction files owned by individual project repositories remain independent.

Protected Linux installations that use Codex image input also require `setfacl` (normally provided by the distribution's `acl` package). Kai uses a read-only named ACL so an image can remain private to the service and its intended `os_user`; if ACL support is unavailable, that image is dropped with a user-visible notice instead of being made world-readable.

For full installation details, see [Getting Started](https://github.com/dcellison/kai/wiki/Getting-Started), [Multi-User Setup](https://github.com/dcellison/kai/wiki/Multi-User-Setup), and [System Architecture](https://github.com/dcellison/kai/wiki/System-Architecture).

## Security Model

Kai has real local authority, so the security model is part of the product rather than an afterthought.

- **Telegram allowlist:** Only configured Telegram user IDs can interact with the bot.
- **Optional TOTP gate:** Time-based one-time passwords can protect the chat surface after idle timeout.
- **Local execution:** Kai runs on your machine. Conversations do not pass through a Kai-hosted relay.
- **Path confinement:** File exchange is constrained to allowed workspace and file-storage paths.
- **Protected backend registry:** Protected installs resolve backend identifiers through admin-owned `/etc/kai/backends.yaml`; executable paths and allowed model surfaces are installation state, not user input.
- **Service proxy:** Third-party API keys live in server-side config and are injected only for services explicitly allowed by the protected runtime profile (or compatibility configuration outside a protected install); keys are never placed in conversation context.
- **GitHub operation boundary:** PR review and issue triage run only for repositories explicitly authorized to that user in admin-controlled `users.yaml`. Protected installs require the user's stored GitHub token, and notification subscriptions cannot grant operation access.
- **Per-user isolation:** Users have separate history, files, workspaces, jobs, settings, and agent subprocesses.
- **Principal-bound internal API:** Agent API credentials resolve to a fixed user and explicit scopes in the outer service; request data cannot select another principal.
- **Separated webhook credentials:** GitHub, generic, and Telegram ingress use distinct secrets that are not exposed to persistent agent subprocesses.
- **Optional OS isolation:** A user's backend subprocess can run under a dedicated OS account through generated sudoers rules.

The former shared `WEBHOOK_SECRET` is no longer supported and never
authenticates a runtime route. `make config` omits it from regenerated
configuration, and `make install` strips it from older artifacts before writing
the deployed environment. If an older artifact lacks either named replacement,
installation fails before stopping Kai and asks for a one-time `make config`.
GitHub and generic callers must use their dedicated named secrets.

Run `make install-status` to inspect the authoritative deployed migration
state in `/etc/kai/env`. The command uses sudo because that file is root-only;
it reports only whether the unsupported and named variables are configured,
never their values. It also labels the separate `install.conf` artifact state so
configuration drift is visible rather than mistaken for deployed truth.

The current remediation status and compatibility exceptions are tracked in
[Security Remediation Status](SECURITY_REMEDIATION_STATUS.md).

See [TOTP Authentication](https://github.com/dcellison/kai/wiki/TOTP-Authentication), [GitHub Notification Routing](https://github.com/dcellison/kai/wiki/GitHub-Notification-Routing), and [Exposing Kai to the Internet](https://github.com/dcellison/kai/wiki/Exposing-Kai-to-the-Internet) for the detailed operational docs.

## Common Workflows

- Send a normal Telegram message to have Kai work in the current workspace.
- Use `/workspace <name>` or `/workspaces` to move between projects.
- Use `/models` or `/model <name>` to change the active model.
- Use `/memory`, `/memory search <query>`, and `/memory stats` to inspect durable memory.
- Ask Kai to remind you later, run a recurring check, or monitor a condition.
- Subscribe a GitHub repo so pushes, PRs, issues, comments, and reviews can reach Kai.
- Enable PR review or issue triage per user when you want background GitHub automation.
- Send files directly in Telegram so the agent can inspect or transform them locally.
- Use `/help` in Telegram for the current command reference.

## Documentation

Most operational documentation lives in the wiki so it can grow without turning the README into a control panel manual.

- [Changelog](CHANGELOG.md)
- [Wiki Home](https://github.com/dcellison/kai/wiki)
- [Getting Started](https://github.com/dcellison/kai/wiki/Getting-Started)
- [System Architecture](https://github.com/dcellison/kai/wiki/System-Architecture)
- [Multi-User Setup](https://github.com/dcellison/kai/wiki/Multi-User-Setup)
- [Scheduling and Conditional Jobs](https://github.com/dcellison/kai/wiki/Scheduling-and-Conditional-Jobs)
- [PR Review Agent](https://github.com/dcellison/kai/wiki/PR-Review-Agent)
- [Issue Triage Agent](https://github.com/dcellison/kai/wiki/Issue-Triage-Agent)
- [GitHub Notification Routing](https://github.com/dcellison/kai/wiki/GitHub-Notification-Routing)
- [Exposing Kai to the Internet](https://github.com/dcellison/kai/wiki/Exposing-Kai-to-the-Internet)
- [Voice Setup](https://github.com/dcellison/kai/wiki/Voice-Setup)
- [TOTP Authentication](https://github.com/dcellison/kai/wiki/TOTP-Authentication)

## Development

```bash
make setup      # Install in editable mode with dev tools
npm ci --prefix workshop-client # Install React client development dependencies
make lint       # Run ruff
make format     # Format with ruff
make check      # Lint and format check
make typecheck  # Run Pyright on the maintained typed baseline
make client-check # Type-check/test React and verify its packaged assets
make audit-deps # Report known vulnerabilities in installed dependencies
make check-install-constraints # Dry-run install dependency resolution with constraints
make module-sizes # Report large Python modules for decomposition planning
make test       # Run pytest
make run        # Start Kai locally
```

Pull requests are currently restricted to collaborators while the architecture is moving quickly. Issues, bug reports, design feedback, and focused proposals are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Kai is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
