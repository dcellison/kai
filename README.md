# Kai

[![CI](https://img.shields.io/github/actions/workflow/status/dcellison/kai/ci.yml?branch=main&label=CI)](https://github.com/dcellison/kai/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue)](https://python.org)
[![License](https://img.shields.io/github/license/dcellison/kai)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/dcellison/kai?label=version)](https://github.com/dcellison/kai/releases)

Kai is an AI agent, not a chatbot. It manages persistent agent subprocesses running on your hardware - [Claude Code](https://docs.anthropic.com/en/docs/claude-code) by default, with [Goose](https://block.github.io/goose/), [OpenAI Codex CLI](https://github.com/openai/codex), or [OpenCode](https://opencode.ai/) as alternative backends - and connects them to Telegram as a control surface. Shell, filesystem, git, web search, scheduling - the agent has real access to your system and can take real action on it. It reviews PRs when code is pushed, triages issues when they're opened, monitors conditions on a schedule, and operates across any project on your machine. Multiple users can share a single Kai instance, each with their own isolated subprocess, workspace, conversation history, and optional OS-level process separation.

For detailed guides on setup, architecture, and optional features, see the **[Wiki](https://github.com/dcellison/kai/wiki)**.

## Architecture

Kai is two layers: an outer Python application that handles Telegram, HTTP, and scheduling, and one or more inner agent subprocesses that do the thinking and acting. The outer process programs against an `AgentBackend` ABC (`backend.py`), so it manages lifecycle, authentication, and transport identically regardless of which backend is running. Claude Code is the default; Goose ACP, OpenAI Codex CLI, and OpenCode ACP are alternatives. All four follow the same lifecycle: one subprocess per user, lazy creation, idle eviction. Resource usage scales with active users, not registered ones. The two ACP-based backends (Goose and OpenCode) share a transport layer in `acp.py`; each adds a thin adapter for the harness-specific argv, env, and notification shapes.

This is what separates Kai from API-wrapper bots that send text to a model endpoint and relay the response. The inner agent is a full agentic runtime - it reads files, runs shell commands, searches the web, writes and commits code, and maintains context across a session. Claude Code and Goose each bring their own model ecosystem. Kai gives the runtime a durable home, a scheduling system, event-driven inputs, persistent memory, and a security model designed around the fact that it has all of this power.

Everything runs locally. Conversations never transit a relay server. Voice transcription and synthesis happen on-device. API keys are proxied through an internal service layer so they never appear in conversation context. There is no cloud component between you and your machine.

## Security model

Giving an AI agent shell access is a real trust decision. Kai's approach is layered defense - each layer independent, so no single failure is catastrophic:

- **Telegram auth** - only explicitly whitelisted user IDs can interact. Unauthorized messages are silently dropped before reaching any handler.
- **TOTP gate** (optional) - two-factor authentication via time-based one-time passwords. After a configurable idle timeout, Kai requires a 6-digit authenticator code before processing anything. The secret lives in a root-owned file (mode 0600) that the bot process cannot read directly; it verifies codes through narrowly-scoped sudoers rules. Even if someone compromises your Telegram account, they can't use your assistant without your authenticator device. Rate limiting with disk-persisted lockout protects against brute force.
- **Process isolation** - authentication state lives in the bot's in-memory context, not in the filesystem or conversation history. The inner agent process cannot read, manipulate, or bypass the auth gate.
- **Path confinement** - file exchange operations are restricted to the active workspace via `Path.relative_to()`. Traversal attempts are rejected. The send-file API also blocks access to history and memory directories.
- **Service proxy** - external API keys live in server-side config (`services.yaml`) and are injected at request time. The agent calls APIs through a local proxy endpoint; the keys never enter the conversation. Per-service SSRF controls prevent services from being used as open HTTP proxies.
- **Multi-user isolation** - each user's data is namespaced by chat ID: separate conversation history, workspace state, scheduled jobs, and file storage. When `os_user` is configured in `users.yaml`, the inner agent subprocess runs as a dedicated OS account via `sudo -u`, creating a hard process-level boundary between users and between the bot and the AI.

Setup for TOTP requires the optional dependency group and root access:

```bash
pip install -e '.[totp]'             # adds pyotp and qrcode
sudo python -m kai totp setup        # generate secret, display QR code, confirm
```

For the full architecture, see [System Architecture](https://github.com/dcellison/kai/wiki/System-Architecture). For TOTP details, see [TOTP Authentication](https://github.com/dcellison/kai/wiki/TOTP-Authentication).

## Features

### Workspaces

Switch the agent between projects on your system with `/workspace <name>`. Names resolve relative to `WORKSPACE_BASE` (set in `.env`). Identity and memory carry over from the home workspace, so Kai retains full context regardless of what it's working on. Create new workspaces with `/workspace new <name>`. Absolute paths are not accepted - all workspaces must live under the configured base directory.

Per-workspace configuration is supported via `workspaces.yaml` (or `/etc/kai/workspaces.yaml` for protected installations). Each workspace can override the model, budget, timeout, environment variables, and system prompt. See `templates/workspaces.yaml` for the full format.

### Multi-user

A single Kai instance can serve multiple Telegram users, each fully isolated. `users.yaml` is mandatory; the daemon fails closed at startup if it cannot find or parse it. The canonical path depends on the deployment mode:

- **Protected install** (`sudo make install`): `/etc/kai/users.yaml` (root-owned, mode 0600).
- **Single-user install** (`make config` then `make run`): `${XDG_CONFIG_HOME:-$HOME/.config}/kai/users.yaml` (operator-owned, mode 0600 in a 0700 parent).

`make config` writes the correct location for the deployment mode you select at the first prompt.

```yaml
users:
  - telegram_id: 123456789
    name: alice
    role: admin           # receives webhook notifications (GitHub, generic)
    github: alice-dev     # routes GitHub events to this user
    os_user: alice        # subprocess runs as this OS account
    home_workspace: /home/alice/workspace
    max_budget: 15.0      # default budget for this user (BUDGET_CEILING is the global ceiling)
    pr_review: true       # enable automatic PR review for this user
    issue_triage: true    # enable automatic issue triage for this user
    github_notify_chat_id: -100123456789   # route GitHub notifications to a group
```

Each user gets:

- **Own agent subprocess** - created lazily on first message, evicted after idle timeout (`AGENT_IDLE_TIMEOUT`, default 30 minutes). No shared conversation state.
- **Isolated data** - conversation history, workspace settings, scheduled jobs, and file uploads are all namespaced by user. One user cannot see or affect another's state.
- **Optional OS-level separation** - set `os_user` to run that user's agent process as a dedicated system account via `sudo -u`. Requires a sudoers rule (the install script generates one automatically).
- **Per-user home workspace** - each user can have their own default workspace directory.
- **Role-based routing** - admins receive unattributed webhook events (GitHub pushes, generic webhooks). Regular users interact only through Telegram messages.

Per-user GitHub agent toggles (`pr_review`, `issue_triage`) and notification routing (`github_notify_chat_id`) live in `users.yaml` and can be overridden at runtime via `/github reviews on|off`, `/github triage on|off`, and `/github notify <chat_id>`. See the [Multi-User Setup](https://github.com/dcellison/kai/wiki/Multi-User-Setup) wiki page for the full field reference.

### Memory

Three layers of persistent context give the agent continuity across sessions:

1. **Identity** (`DATA_DIR/home/<chat_id>/.claude/CLAUDE.md`) - voice, rules, and operational guidelines. Lives in each operator's per-user home workspace, seeded from `templates/.claude/CLAUDE.md` at install time and lazily on first message for users added later. When the agent switches to a foreign workspace, Kai injects it so behavior stays consistent. In the home workspace, the agent reads it natively.
2. **Home memory** (`DATA_DIR/memory/<chat_id>/MEMORY.md`) - per-user personal memory, always injected regardless of current workspace. Proactively updated by Kai. Each user has their own file under `memory/<chat_id>/`, scoped by Telegram chat_id so memories stay private.
3. **Conversation history** (`DATA_DIR/history/<chat_id>/`) - JSONL logs, one file per day per user. Searchable for past conversations.

Workspaces can also define a system prompt via `workspaces.yaml` for workspace-specific instructions. See [System Architecture](https://github.com/dcellison/kai/wiki/System-Architecture).

#### Memory backend selection

Semantic memory extraction (the subprocess that proactively writes facts and episode summaries) routes per-user: each user's effective `agent_backend` selects the reasoner, and the model comes from the project's `MODEL_REGISTRY` for that `(role, backend)` pair. Claude-effective users get the claude reasoner with the claude-registry model; codex-effective users get the codex reasoner with the codex-registry model; opencode-effective users get the opencode reasoner with the opencode-registry model (a `provider/model` string resolved at runtime by `opencode` against the operator's `opencode auth login` state). There is no per-deployment override for the memory reasoner or model; an operator who wants a different model edits the registry in `config.py`.

OpenCode users get the same one-shot parity as claude and codex users: PR review, issue triage, memory extraction, episode generation, and behavioral eval all dispatch through `OpenCodeOneShotReasoner`, which spawns a fresh `opencode acp` JSON-RPC subprocess per call and denies any tool-permission request mid-stream. No fall-through to `claude --print` or `codex exec` for opencode users on any one-shot site.

Each backend that runs extraction on this install must have its binary reachable at startup; the bot exits at config-load otherwise with a message naming the offending backend and resolution sequence. Retrieval-only memory (`MEMORY_ENABLED=true` with extraction disabled) requires no extraction binary.

To verify a memory configuration end-to-end without writing to the store, run `python -m kai.smoke.memory` from the install directory. Pass `--user-id <chat_id>` to drive the smoke under that user's effective backend (otherwise the global `AGENT_BACKEND` is used). Pass `--os-user <name>` when the effective backend resolves to codex or opencode and the target user is not the bot process user; both reasoners refuse to spawn under sudo without one. The smoke prints the resolved binary, the argv that ran, and any extracted facts.

### Scheduled jobs

Reminders and recurring agent jobs with one-shot, daily, and interval schedules. Ask naturally ("remind me at 3pm") or use the HTTP API (`POST /api/schedule`). Agent jobs run as full agent sessions - Kai can check conditions, search the web, run commands, and report back on a schedule. Auto-remove jobs support monitoring use cases where the agent watches for a condition and deactivates itself when it's met. See [Scheduling and Conditional Jobs](https://github.com/dcellison/kai/wiki/Scheduling-and-Conditional-Jobs).

### PR Review Agent

When code is pushed to a pull request, Kai automatically reviews it. A one-shot agent subprocess analyzes the diff, checks for bugs, style issues, and spec compliance, and posts a review comment directly on the PR. If you push fixes, it reviews again - and checks its own prior comments so it doesn't nag about things you already addressed. See [PR Review Agent](https://github.com/dcellison/kai/wiki/PR-Review-Agent).

### Issue Triage Agent

When a new issue is opened, Kai triages it automatically. A one-shot agent subprocess reads the issue, applies labels (creating them if they don't exist), checks for duplicates and related issues, assigns it to a project board if appropriate, posts a triage summary comment, and sends you a Telegram notification. See [Issue Triage Agent](https://github.com/dcellison/kai/wiki/Issue-Triage-Agent).

Both agents are fire-and-forget background tasks that run independently of your chat session. They use separate agent processes, so a review or triage can happen while you're mid-conversation. Opt-in per-user via `pr_review` / `issue_triage` in `users.yaml`, or toggle at runtime with `/github reviews on|off` and `/github triage on|off`.

### GitHub notification routing

GitHub event notifications (pushes, PRs, issues, comments, reviews) can be routed to a separate Telegram group via `github_notify_chat_id` in `users.yaml` (or `/github notify <chat_id>` at runtime), keeping your primary DM clean for conversation. Agent output (review comments, triage summaries) also routes to the group. See [GitHub Notification Routing](https://github.com/dcellison/kai/wiki/GitHub-Notification-Routing).

### Webhooks

An HTTP server receives external events and routes them to the agent. GitHub webhooks (pushes, PRs, issues, comments, reviews) are validated via HMAC-SHA256. A generic endpoint (`POST /webhook`) accepts JSON from any service - CI pipelines, monitoring alerts, deployment hooks, anything that can POST JSON. See [Exposing Kai to the Internet](https://github.com/dcellison/kai/wiki/Exposing-Kai-to-the-Internet).

### File exchange

Send any file type directly in chat - photos, documents, PDFs, archives, anything. Files are saved to per-user directories under the data directory with timestamped names, and the agent gets the path so it can work with them via shell tools. Kai can also send files back to you through the internal API. Images render inline; everything else arrives as a document attachment. Set `FILE_RETENTION_DAYS` to automatically clean up old uploads.

### Streaming responses

Responses stream into Telegram in real time, updating the message every 2 seconds.

### Model switching

Switch models via `/models` (interactive picker) or `/model <name>` (direct). Available models depend on the configured backend: Claude Code offers Opus, Sonnet, and Haiku; Goose offers whatever the user's configured provider supports; Codex offers the Codex CLI's curated model list; OpenCode accepts free-text `provider/model` IDs (for example `anthropic/claude-sonnet-4-6`) that OpenCode resolves at runtime against the credentials in `~/.local/share/opencode/auth.json`. Changing models restarts the session.

### Voice input

Voice notes are transcribed locally using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and forwarded to the agent. Requires `ffmpeg` and `whisper-cpp`. Disabled by default - set `VOICE_ENABLED=true` after installing dependencies. See the [Voice Setup](https://github.com/dcellison/kai/wiki/Voice-Setup) wiki page.

### Voice responses (TTS)

Text-to-speech via [Piper TTS](https://github.com/rhasspy/piper). Three modes: `/voice only` (voice note, no text), `/voice on` (text + voice), `/voice off` (text only, default). Eight curated English voices. Requires `pip install -e '.[tts]'` and `TTS_ENABLED=true`. See [Voice Setup](https://github.com/dcellison/kai/wiki/Voice-Setup).

### Dual-mode Telegram transport

Kai supports two ways of receiving Telegram updates: **long polling** (default) and **webhooks**. Polling works out of the box behind NAT with zero infrastructure. Set `TELEGRAM_WEBHOOK_URL` in `.env` to switch to webhook mode for lower latency - this requires a tunnel or reverse proxy (see [Exposing Kai to the Internet](https://github.com/dcellison/kai/wiki/Exposing-Kai-to-the-Internet)).

### Crash recovery

If interrupted mid-response, Kai notifies you on restart and asks you to resend your last message.

## Commands

| Command | Description |
|---|---|
| `/new` | Clear session and start fresh |
| `/stop` | Interrupt a response mid-stream |
| `/models` | Interactive model picker |
| `/model <name>` | Switch model (available models depend on backend) |
| `/settings` | Show per-user settings (model, budget, timeout, context window) |
| `/settings <field> <value>` | Change a setting (`model`, `budget`, `timeout`, `context`) |
| `/settings reset [field]` | Clear all overrides, or one field |
| `/workspace` (or `/ws`) | Show current workspace |
| `/workspace <name>` | Switch by name (resolved under `WORKSPACE_BASE`) |
| `/workspace home` | Return to default workspace |
| `/workspace new <name>` | Create a new workspace with git init |
| `/workspace allow <path>` | Add an allowed workspace for your user |
| `/workspace deny <path>` | Remove an allowed workspace for your user |
| `/workspace allowed` | List your allowed workspaces |
| `/workspaces` | Interactive workspace picker |
| `/memory` | Browse remembered facts grouped by tag (inline buttons) |
| `/memory search <query>` | Semantic search over remembered facts |
| `/memory stats` | Count of facts and confidence distribution |
| `/memory forget <tag>` | Delete every fact with the given tag |
| `/memory help` | `/memory` subcommand reference |
| `/github` | Show GitHub notification settings |
| `/github notify <chat_id>` | Route your GitHub notifications to a specific chat |
| `/github reviews on\|off` | Enable or disable the PR review agent for you |
| `/github triage on\|off` | Enable or disable the issue triage agent for you |
| `/github add <owner/repo>` | Subscribe to a repo (auto-registers webhook if token set) |
| `/github remove <owner/repo>` | Unsubscribe from a repo |
| `/github token <token>` | Store GitHub PAT for auto-webhook registration (delete the message after sending) |
| `/voice` | Toggle voice responses on/off |
| `/voice only` | Voice-only mode (no text) |
| `/voice on` | Text + voice mode |
| `/voice <name>` | Set voice |
| `/voices` | Interactive voice picker |
| `/stats` | Show session info, model, and cost |
| `/job` (or `/jobs`) | List scheduled jobs |
| `/job info <id>` | Show job details |
| `/job cancel <id>` | Cancel a scheduled job |
| `/webhooks` | Show webhook server status |
| `/help` | Show available commands |

## Requirements

- Python 3.13+
- One of: [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (default), [Goose CLI](https://block.github.io/goose/), [OpenAI Codex CLI](https://github.com/openai/codex), or [OpenCode CLI](https://opencode.ai/) - installed and authenticated for the backend you select
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- Your Telegram user ID (get it from [@userinfobot](https://t.me/userinfobot))

## Setup

```bash
git clone git@github.com:dcellison/kai.git
cd kai
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp templates/.env .env
```

### Environment variables

Authorization, per-user model selection, per-user OS isolation, per-user GitHub routing, and per-user agent toggles all live in `users.yaml`. The variables below are global resource controls and installation-wide defaults; per-user fields are documented in [Multi-User Setup](https://github.com/dcellison/kai/wiki/Multi-User-Setup).

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | | Bot token from BotFather |
| `AGENT_BACKEND` | No | `claude` | Global agent backend: `claude`, `goose`, `codex`, or `opencode`. Per-user override goes in `users.yaml`. |
| `LLM_PROVIDER` | Non-claude | | Global provider for non-claude backends. Per-user override in `users.yaml`. |
| `DEFAULT_MODEL` | No | `sonnet` | Installation-wide default model. Per-user override in `users.yaml` `model`, or `/settings model`. |
| `AGENT_TIMEOUT_SECONDS` | No | `120` | Installation-wide default per-message timeout. Per-user override in `users.yaml` `timeout`, or `/settings timeout`. |
| `BUDGET_CEILING` | No | `10.0` | Global budget ceiling in USD. Users cannot exceed this via `/settings budget`. Per-user defaults in `users.yaml` `max_budget`. |
| `CLAUDE_MAX_CONTEXT_WINDOW` | No | `0` | Installation-wide default context window in tokens (0 = backend default). Per-user override in `users.yaml` `context_window`, or `/settings context`. |
| `CLAUDE_AUTOCOMPACT_PCT` | No | `80` | Context compression threshold %, Claude Code only. When usage hits this, Claude compresses history. Can only lower the default (~83%), not raise it. |
| `AGENT_MAX_SESSION_HOURS` | No | `0` | Maximum session age in hours before recycling the subprocess (0 = no limit). Applies to every backend. Recommended: 4-8 on memory-constrained machines. |
| `WORKSPACE_BASE` | No | | Installation-wide default workspace base directory. Per-user override in `users.yaml` `workspace_base`. |
| `ALLOWED_WORKSPACES` | No | | Comma-separated extra workspace paths accessible by name. Users also manage their own via `/workspace allow`. |
| `WEBHOOK_PORT` | No | `8080` | HTTP server port for webhooks and scheduling API |
| `WEBHOOK_SECRET` | No | | Secret for webhook validation and scheduling API auth |
| `TELEGRAM_WEBHOOK_URL` | No | | Telegram webhook URL (enables webhook mode; omit for polling) |
| `TELEGRAM_WEBHOOK_SECRET` | No | | Separate secret for Telegram webhook auth (defaults to `WEBHOOK_SECRET`) |
| `PR_REVIEW_COOLDOWN` | No | `300` | Minimum seconds between reviews of the same PR. Machine-wide resource limit. |
| `PR_REVIEW_TIMEOUT_S` | No | `900` | Subprocess timeout for a single PR review, in seconds. |
| `PR_REVIEW_BUDGET_USD` | No | `1.0` | Hard USD ceiling for one PR review subprocess. Informational on the claude backend; enforced on non-claude backends. |
| `SPEC_DIR` | No | `specs` | Spec directory relative to repo root, for branch-name matching in PR reviews |
| `AGENT_IDLE_TIMEOUT` | No | `1800` | Seconds before idle subprocesses are evicted (0 to disable). Applies to every backend. |
| `CLAUDE_EFFORT_LEVEL` | No | `high` | Reasoning effort for inner Claude: `low`, `medium`, `high`, `xhigh`, `max`. |
| `VOICE_ENABLED` | No | `false` | Enable voice message transcription |
| `TTS_ENABLED` | No | `false` | Enable text-to-speech voice responses |
| `TOTP_SESSION_MINUTES` | No | `30` | Minutes before TOTP re-authentication is required |
| `TOTP_CHALLENGE_SECONDS` | No | `120` | Seconds the code entry window stays open |
| `TOTP_LOCKOUT_ATTEMPTS` | No | `3` | Failed TOTP attempts before temporary lockout |
| `TOTP_LOCKOUT_MINUTES` | No | `15` | TOTP lockout duration in minutes |
| `FILE_RETENTION_DAYS` | No | `0` | Days to keep uploaded files before cleanup (0 to disable) |

`BUDGET_CEILING` limits spending in a single session. On the Claude Code backend the flag is omitted on Max-plan OAuth (no real cost is incurred); runaway prevention is handled by the per-session timeout instead. Goose does not currently enforce this limit. The session resets on `/new`, model switch, or workspace switch.

## Running

Run the configuration wizard first; it writes both `users.yaml` (mandatory authorization) and the runtime env file in the location that matches your deployment mode.

```bash
make config
```

The wizard's first prompt picks the deployment mode:

- **`single_user`** - Kai runs from the cloned repo under your own OS account. Secrets go to `PROJECT_ROOT/.env` (mode 0600). `users.yaml` goes to `${XDG_CONFIG_HOME:-$HOME/.config}/kai/users.yaml` (mode 0600 in a 0700 parent). No `sudo` required.
- **`protected`** - Kai runs from a root-owned install at `/opt/kai/`. Secrets go to `/etc/kai/env`, `users.yaml` goes to `/etc/kai/users.yaml`, and a sudoers rule lets the service user read them via `sudo cat`. Requires `sudo make install` after the wizard.

After the wizard:

```bash
make run    # single-user mode
# OR
sudo make install   # protected mode (creates /opt/kai layout; copies secrets)
```

Or manually: `source .venv/bin/activate && python -m kai`

### Running as a service (macOS)

Create `~/Library/LaunchAgents/com.kai.bot.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kai.bot</string>

    <key>ProgramArguments</key>
    <array>
        <string>/path/to/kai/.venv/bin/python</string>
        <string>-m</string>
        <string>kai</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/path/to/kai</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
```

Replace `/path/to/kai` with your actual project path. The `PATH` must include directories for `claude` (or `goose`, depending on your backend), `ffmpeg`, and any other tools Kai shells out to.

```bash
launchctl load ~/Library/LaunchAgents/com.kai.bot.plist
```

Kai will start immediately and restart automatically on login or crash. Logs go to `logs/kai.log` with daily rotation (14 days of history). To stop:

```bash
launchctl unload ~/Library/LaunchAgents/com.kai.bot.plist
```

### Running as a service (Linux)

Create `/etc/systemd/system/kai.service`:

```ini
[Unit]
Description=Kai Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/kai
ExecStart=/path/to/kai/.venv/bin/python -m kai
Restart=always
RestartSec=5
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
```

Replace `YOUR_USERNAME` and `/path/to/kai` with your values. Add any extra directories to `PATH` where `claude`, `ffmpeg`, etc. are installed.

The `network-online.target` dependency ensures systemd waits for network connectivity before starting Kai, preventing DNS failures during boot.

```bash
sudo systemctl enable kai
sudo systemctl start kai
```

Check logs with `tail -f logs/kai.log` or `journalctl -u kai -f`. To stop:

```bash
sudo systemctl stop kai
```

## Project Structure

```
kai/
├── src/kai/                  # Source package
│   ├── __init__.py           # Version
│   ├── __main__.py           # python -m kai entry point
│   ├── backend.py            # Agent backend ABC and shared context injection
│   ├── main.py               # Async startup and shutdown
│   ├── bot.py                # Telegram handlers, commands, message routing
│   ├── claude.py             # Persistent Claude Code subprocess management
│   ├── config.py             # Environment and per-workspace config loading
│   ├── acp.py                # Shared ACP transport (Goose + OpenCode base layer)
│   ├── goose.py              # Goose ACP backend (adapter over acp.py)
│   ├── opencode.py           # OpenCode ACP backend (adapter over acp.py)
│   ├── sessions.py           # SQLite session, job, and settings storage
│   ├── cron.py               # Scheduled job execution (APScheduler)
│   ├── webhook.py            # HTTP server: GitHub/generic webhooks, scheduling API
│   ├── history.py            # Conversation history (read/write JSONL logs)
│   ├── pool.py               # Per-user agent subprocess pool (lazy creation, idle eviction)
│   ├── locks.py              # Per-chat async locks and stop events
│   ├── install.py            # Protected installation tooling
│   ├── totp.py               # TOTP verification, rate limiting, and CLI
│   ├── review.py             # PR review agent (one-shot agent subprocess)
│   ├── triage.py             # Issue triage agent (one-shot agent subprocess)
│   ├── services.py           # External service proxy for third-party APIs
│   ├── transcribe.py         # Voice message transcription (ffmpeg + whisper-cpp)
│   ├── tts.py                # Text-to-speech synthesis (Piper TTS + ffmpeg)
│   ├── prompt_utils.py       # Shared prompt construction utilities
│   ├── telegram_utils.py     # Telegram-specific helper functions
│   └── workspace_utils.py    # Workspace path resolution and validation
├── tests/                    # Test suite
├── templates/                # Tracked seed files copied into the install on first run
│   ├── .claude/              # Identity and memory templates (CLAUDE.md, MEMORY.md, PREFERENCES.md)
│   ├── config/               # Static config templates (goose-config.yaml)
│   ├── .env                  # Environment variable template
│   ├── services.yaml         # External service config template
│   ├── users.yaml            # Multi-user config template
│   └── workspaces.yaml       # Per-workspace config template
├── kai.db                    # SQLite database (gitignored, created at runtime)
├── logs/                     # Daily-rotated log files (gitignored)
├── models/                   # Whisper and Piper model files (gitignored)
├── services.yaml             # External service configs (gitignored)
├── pyproject.toml            # Package metadata and dependencies
├── Makefile                  # Common dev commands
├── .env                      # Environment variables (gitignored, copy from templates/.env)
└── LICENSE                   # Apache 2.0
```

## Development

```bash
make setup      # Install in editable mode with dev tools
make lint       # Run ruff linter
make format     # Auto-format with ruff
make check      # Lint + format check (CI-friendly)
make test       # Run test suite
make run        # Start the bot
```

## Production deployment

The protected install mode separates source, data, and secrets across protected directories:

```bash
python -m kai install config       # Interactive Q&A; pick `protected` at the first prompt (no sudo)
sudo python -m kai install apply   # Creates /opt layout, migrates data (root)
python -m kai install status       # Shows current installation state (no sudo)
```

This creates a split layout:

- `/opt/kai/` - read-only source and venv (root-owned)
- `/var/lib/kai/` - writable runtime data: database, logs, files (service-user-owned)
- `/etc/kai/` - secrets: env file, `users.yaml`, service configs, TOTP (root-owned, mode 0600)

The install module handles directory creation, source copying, venv setup, secret deployment, sudoers rules, data migration, service definition generation, and service lifecycle (stop/start). Use `--dry-run` to preview changes without applying them.

The service user reads secrets via narrowly-scoped sudoers rules (`sudo cat` on specific files only). The inner agent process cannot read secrets or modify source code.

`make config` also supports single-user installs (pick `single_user` at the first prompt). In that mode `.env` and `users.yaml` live under the operator's account; `make install` is a no-op and Kai starts via `make run` directly from the cloned repo.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
