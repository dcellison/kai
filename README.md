# Kai

[![CI](https://img.shields.io/github/actions/workflow/status/dcellison/kai/ci.yml?branch=main&label=CI)](https://github.com/dcellison/kai/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue)](https://python.org)
[![License](https://img.shields.io/github/license/dcellison/kai)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/dcellison/kai?label=version)](https://github.com/dcellison/kai/releases)

**A workshop on your own hardware where people and AI agents work together.**

Kai is a self-hosted service that gives coding agents a durable home: persistent processes with shell, filesystem, git, and web access to the machine you run it on. You reach it through the Workshop, a browser client where people and agents share channels, or through Telegram from your phone. Agents remember you between sessions, review your pull requests when you push, triage issues when they open, run scheduled jobs while you sleep, and work in any project directory you authorize.

Everything runs locally. Conversations never transit a relay server. Your machine, your data, your rules.

New here? The [wiki](https://github.com/dcellison/kai/wiki) has the full guides; the [Quick Start](#quick-start) below gets you running.

## Why Kai Exists

Most AI coding tools are a terminal session or a hosted chat tab: you open them, work, and everything evaporates when you close them. Kai is built for a different operating model, an always-on local service where agents are long-lived participants rather than disposable sessions.

- **Agents are teammates, not tabs.** Each agent is a persistent process with its own identity, memory, and conversation history that survive restarts, upgrades, and workspace switches.
- **Real authority.** Agents run on hardware you control, against your actual repositories, shell, and tools, with a security model built around the fact that they can take real action.
- **A shared room.** The Workshop puts people and agents in the same channels, with threads, mentions, reactions, file artifacts, and live streaming runs you can inspect tool call by tool call.
- **Work happens without you.** PR review, issue triage, reminders, recurring jobs, condition monitors, and webhook reactions all run in the background and land in your notification feed.
- **No vendor holds the keys.** Five interchangeable agent backends mean the provider is a configuration choice, not a structural dependency. When one has a bad day, route around it.
- **One instance, many people.** Multiple users share a single install with isolated conversations, memory, files, workspaces, and optionally separate OS accounts per agent.

## The Workshop

The Workshop is Kai's native surface: a real-time browser client served by the same local process.

- **Channels, direct messages, notification feeds**, with threads, mentions, reactions, unread tracking, and file artifacts that render inline.
- **Agent runs stream live.** A working agent shows a status banner and a writing preview; the run inspector shows every tool call as it happens, and Stop actually stops it.
- **Build your own agents in the browser.** Give an agent a handle, a purpose, instructions, and declared capabilities. Definitions are versioned and immutable; draft a revision, activate it when ready, roll back by activating an older one.
- **Shared definitions, private conversations.** Anyone can enable an agent someone else built, but each person gets their own private lane and the agent keeps a separate memory of each person. Same instructions, nothing else crosses over.
- **Agents delegate to agents.** An agent with the delegation capability can hand bounded sub-tasks to other agents in a shared channel, under server-enforced limits, with every delegated run visible like any other.
- **A memory explorer.** Browse, search, and curate what Kai remembers about you.

Telegram is the optional second surface: chat with your agents from your phone, with voice notes, file exchange, streaming replies, and slash commands. Enable either surface or both; a person provisioned on one keeps their history when the other is linked later.

## Choose Your Backend

Each Kai backend is a full coding harness with its own protocol, tools, and authentication. Kai normalizes lifecycle and routing around those harnesses and lets each user, even each channel agent, run on a different one. Switch models mid-conversation with a command or a click.

| Backend | Runtime | Model Selection Shape | Notes |
|---|---|---|---|
| Claude Code | `claude` CLI | Claude aliases and full model IDs | The default. Uses Claude Code's local authentication. |
| OpenAI Codex CLI | `codex` CLI | Codex CLI model IDs | Uses Codex's own model catalog. |
| Goose | `goose acp` | Provider-native model IDs | ACP backend; provider selected through Goose configuration. |
| OpenCode | `opencode acp` | `provider/model` IDs | ACP backend; model resolution owned by OpenCode. |
| Pi | `pi --mode rpc` | `provider/model[:thinking]` IDs | JSONL RPC backend; bounded one-shot tasks disable tools. |

Between them the backends cover providers like OpenAI, Google, DeepSeek, OpenRouter, GitHub Copilot, and local models via Ollama. Only the backends you actually select need to be installed and authenticated. [Multiple Backends](https://github.com/dcellison/kai/wiki/Multiple-Backends) explains the architecture and the case for provider diversity.

## Core Capabilities

| Capability | What Kai Does |
|---|---|
| Repo-aware coding | Runs agents inside local workspaces with shell, filesystem, git, and web access. |
| Workspaces | Switches between projects by name, with per-workspace model, environment, and prompt settings. |
| Memory | Extracts facts and episode summaries into a local vector store and recalls them by relevance, scoped per person and per project. |
| Scheduling | Runs reminders, recurring jobs, and condition monitors that remove themselves when the condition fires. |
| GitHub automation | Reviews PRs, triages issues, routes notifications, and reacts to webhook events. |
| File exchange | Accepts uploads from any surface, exposes authorized local paths to agents, and publishes results back. |
| Voice | Transcribes voice notes locally with whisper.cpp and can answer aloud with Piper text-to-speech. |
| External services | Proxies third-party APIs (search, weather, notifications) so keys never enter conversation context. |
| Multi-user operation | Isolates principals, runtime profiles, workspaces, files, history, jobs, settings, and OS accounts. |

## How It Works

```text
People (browser)         People (Telegram)        GitHub / webhooks
      |                        |                        |
      v                        v                        v
Workshop client API      Telegram adapter         webhook ingress
      \                        |                        /
       +--------- canonical core (kai.db) -------------+
       |  append-only event store -> projections       |
       |  principals, channels, runs, delivery outbox  |
       +----------------------+------------------------+
                              |
                              v
                runtime pool (one lane per channel agent)
                Claude Code and other backends, stream-json
```

At the center is an event-sourced core: one SQLite database holding an append-only event log, with everything else (channels, messages, runs, memory, delivery) derived from it. That buys properties most chat bots never have. Derived state is disposable and rebuilds from the log. Commands are durably acknowledged before execution, so a dropped connection never loses a message. Execution grants and deliveries are fenced with epochs, so crashes and restarts cannot double-run a task or double-send a reply. Delivery goes through a durable outbox; if Telegram is down, the conversation is already recorded and delivery retries under lease.

Agents execute as backend subprocesses speaking stream-json, created lazily per channel agent and evicted when idle, so resource use follows active conversations. The whole service is one Python process on one event loop, running as a LaunchDaemon on macOS or a systemd service on Linux.

[System Architecture](https://github.com/dcellison/kai/wiki/System-Architecture) has the full tour.

## Memory

Kai separates rules (identity files injected every session), facts (what it knows about you), and conversation history. In semantic memory mode, facts live in an embedded local vector store: after each exchange, a one-shot extractor writes structured facts and episode summaries, and every message triggers a relevance-ranked recall scoped to the person and the active project. No external services, no Docker, no open ports; the embedding model runs on your machine. A curated markdown mode is available instead, with a migration path between the two. See [Memory](https://github.com/dcellison/kai/wiki/Memory).

## Quick Start

Requirements:

- Python 3.13+
- At least one supported agent backend installed and authenticated
- A Telegram bot token only if enabling the optional Telegram adapter
- Sudo 1.9.3+ for protected multi-user installs

Install for local development:

```bash
git clone git@github.com:dcellison/kai.git
cd kai
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make config
```

`make config` discovers the backend CLIs installed on the machine and walks you through client mode (Workshop, Telegram, or both), backends, and deployment style. On a fresh Workshop-only configuration it provisions your admin account and prints a one-time browser enrollment token; the token is shown once and only its hash is stored.

For a single-user deployment, start Kai from the checkout and open the Workshop:

```bash
make run
# then visit http://127.0.0.1:8080/workshop/
```

For a protected multi-user deployment, preview and apply the staged install:

```bash
make DRY_RUN=1 install
make install
make install-status
```

`make install` invokes `sudo` internally and separates source, data, and secrets into protected system directories, generates the admin-owned backend registry, and can run each person's agent under its own OS account. `make install-status` reports deployed state without exposing secrets.

The wiki covers every step in depth: [Getting Started](https://github.com/dcellison/kai/wiki/Getting-Started) for your first conversation, [Configuration Wizard](https://github.com/dcellison/kai/wiki/Configuration-Wizard) for every prompt explained, [Installing Agent Binaries](https://github.com/dcellison/kai/wiki/Installing-Agent-Binaries), [Multi-User Setup](https://github.com/dcellison/kai/wiki/Multi-User-Setup), and [Protected Installation](https://github.com/dcellison/kai/wiki/Protected-Installation).

## Security Model

Kai has real local authority, so the security model is part of the product rather than an afterthought.

- **Local execution.** Kai runs on your machine, loopback-only by default. Conversations do not pass through a Kai-hosted relay.
- **Authenticated surfaces.** Workshop access uses single-use enrollment tokens redeemed for hashed session credentials; Telegram uses a per-user allowlist with an optional TOTP gate.
- **Per-principal isolation.** Separate history, memory, files, workspaces, jobs, settings, and agent subprocesses per person; every read and command path checks one authorization authority.
- **OS-level separation.** In protected installs, the daemon runs as a service account and each persistent agent can run under its own OS user through generated sudoers rules.
- **Protected backend registry.** Backend executable paths and allowed model surfaces are admin-owned installation state, never user input.
- **Key custody.** Third-party API keys live in server-side configuration and are injected only for explicitly allowed services; they never appear in conversation context.
- **Bounded ingress.** GitHub, generic, and Telegram webhooks use distinct named secrets, stay loopback-only behind your tunnel, and webhook payload text is wrapped in untrusted-content delimiters before any agent sees it.
- **Path confinement.** File exchange is constrained to allowed workspace and storage paths, with size limits and per-principal access checks on every download.

Remediation history and compatibility exceptions are tracked in [Security Remediation Status](SECURITY_REMEDIATION_STATUS.md). For operations, see [TOTP Authentication](https://github.com/dcellison/kai/wiki/TOTP-Authentication) and [Exposing Kai to the Internet](https://github.com/dcellison/kai/wiki/Exposing-Kai-to-the-Internet).

## Documentation

The [wiki](https://github.com/dcellison/kai/wiki) is the operational reference. A map:

- **Workshop:** [Collaboration Basics](https://github.com/dcellison/kai/wiki/Workshop-Collaboration-Basics) · [Agents](https://github.com/dcellison/kai/wiki/Workshop-Agents) · [Memory Explorer](https://github.com/dcellison/kai/wiki/Workshop-Memory-Explorer) · [Settings](https://github.com/dcellison/kai/wiki/Workshop-Settings) · [Operator Guide](https://github.com/dcellison/kai/wiki/Workshop-Operator-Guide)
- **Setup:** [Getting Started](https://github.com/dcellison/kai/wiki/Getting-Started) · [Configuration Wizard](https://github.com/dcellison/kai/wiki/Configuration-Wizard) · [Multiple Backends](https://github.com/dcellison/kai/wiki/Multiple-Backends) · [Installing Agent Binaries](https://github.com/dcellison/kai/wiki/Installing-Agent-Binaries) · [Multi-User Setup](https://github.com/dcellison/kai/wiki/Multi-User-Setup) · [Protected Installation](https://github.com/dcellison/kai/wiki/Protected-Installation) · [Exposing Kai to the Internet](https://github.com/dcellison/kai/wiki/Exposing-Kai-to-the-Internet) · [Voice Setup](https://github.com/dcellison/kai/wiki/Voice-Setup) · [Browser Automation](https://github.com/dcellison/kai/wiki/Browser-Automation)
- **Features:** [PR Review Agent](https://github.com/dcellison/kai/wiki/PR-Review-Agent) · [Issue Triage Agent](https://github.com/dcellison/kai/wiki/Issue-Triage-Agent) · [Scheduling and Conditional Jobs](https://github.com/dcellison/kai/wiki/Scheduling-and-Conditional-Jobs) · [Workspaces](https://github.com/dcellison/kai/wiki/Workspaces) · [Memory](https://github.com/dcellison/kai/wiki/Memory) · [External Services](https://github.com/dcellison/kai/wiki/External-Services) · [GitHub Notification Routing](https://github.com/dcellison/kai/wiki/GitHub-Notification-Routing) · [Webhook Examples](https://github.com/dcellison/kai/wiki/Webhook-Examples)
- **Design:** [System Architecture](https://github.com/dcellison/kai/wiki/System-Architecture) · [Agent Context Sequence](https://github.com/dcellison/kai/wiki/Agent-Context-Sequence) · [Testing](https://github.com/dcellison/kai/wiki/Testing)
- **Reference:** [Slash Commands](https://github.com/dcellison/kai/wiki/Slash-Commands) · [Troubleshooting](https://github.com/dcellison/kai/wiki/Troubleshooting) · [Changelog](CHANGELOG.md)

## Development

```bash
make setup      # Install in editable mode with dev tools
npm ci --prefix workshop-client # Install React client development dependencies
make lint       # Run ruff
make format     # Format with ruff
make check      # Lint and format check
make typecheck  # Run Pyright on the maintained typed baseline
make client-check # Type-check/test React and verify its packaged assets
make workshop-dev # Serve Workshop with hot reload and proxy the installed API
make audit-deps # Report known vulnerabilities in installed dependencies
make check-install-constraints # Dry-run install dependency resolution with constraints
make module-sizes # Report large Python modules for decomposition planning
make test       # Run pytest
make run        # Start Kai locally
```

The Workshop client is a React application in `workshop-client/`, but operators never need Node: the built bundle is committed and served directly, and CI fails if the committed output is stale. For rapid UI iteration, `make workshop-dev` runs a Vite dev server against a running install; see the [Workshop Operator Guide](https://github.com/dcellison/kai/wiki/Workshop-Operator-Guide) for the details.

Client-only pull requests run the client lane; backend or mixed changes run the complete Python and client validation, and every push to `main` runs the full suite. Unknown paths fail closed to the full lane.

Pull requests are currently restricted to collaborators while the architecture is moving quickly. Issues, bug reports, design feedback, and focused proposals are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Kai is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
