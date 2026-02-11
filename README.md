# Kai

A personal AI assistant accessed via Telegram, powered by [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Kai is a Telegram gateway to a persistent Claude Code process. Messages are forwarded to Claude with full tool access (shell, files, web search), and responses stream back in real time. Everything runs locally — conversations and credentials never leave your machine.

## Requirements

- Python 3.13+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- Your Telegram user ID (get it from [@userinfobot](https://t.me/userinfobot))

## Setup

```bash
git clone git@github.com:dcellison/kai.git
cd kai
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | | Bot token from BotFather |
| `ALLOWED_USER_IDS` | Yes | | Comma-separated Telegram user IDs |
| `CLAUDE_MODEL` | No | `sonnet` | Default model (`opus`, `sonnet`, or `haiku`) |
| `CLAUDE_TIMEOUT_SECONDS` | No | `120` | Per-message timeout |
| `CLAUDE_MAX_BUDGET_USD` | No | `10.0` | Session budget cap |
| `WEBHOOK_PORT` | No | `8080` | HTTP server port for webhooks and scheduling API |
| `WEBHOOK_SECRET` | No | | Secret for webhook validation and scheduling API auth |
| `VOICE_ENABLED` | No | `false` | Enable voice message transcription |
| `TTS_ENABLED` | No | `false` | Enable text-to-speech voice responses |

`CLAUDE_MAX_BUDGET_USD` limits how much work Claude can do in a single session via Claude Code's `--max-budget-usd` flag. On Pro/Max plans this is purely a runaway prevention mechanism (no per-token charges). The session resets on `/new`, model switch, or workspace switch.

## Running

```bash
make run
```

Or manually: `source .venv/bin/activate && python -m kai`

For running as a background service, see the wiki: [macOS (launchd)](https://github.com/dcellison/kai/wiki/Architecture) or Linux (systemd).

## Commands

| Command | Description |
|---|---|
| `/new` | Clear session and start fresh |
| `/stop` | Interrupt a response mid-stream |
| `/models` | Interactive model picker |
| `/model <name>` | Switch model (`opus`, `sonnet`, `haiku`) |
| `/workspace` | Show current workspace |
| `/workspace <name>` | Switch by name or absolute path |
| `/workspace home` | Return to default workspace |
| `/workspace base <path>` | Set the projects directory for short-name resolution |
| `/workspace new <name>` | Create a new workspace with git init |
| `/workspaces` | Interactive workspace picker |
| `/voice` | Toggle voice responses on/off |
| `/voice only` | Voice-only mode (no text) |
| `/voice on` | Text + voice mode |
| `/voice <name>` | Set voice |
| `/voices` | Interactive voice picker |
| `/stats` | Show session info, model, and cost |
| `/jobs` | List active scheduled jobs |
| `/canceljob <id>` | Cancel a scheduled job |
| `/webhooks` | Show webhook server status |
| `/help` | Show available commands |

## Features

### Streaming responses

Responses stream into Telegram in real time, updating the message every 2 seconds.

### Model switching

Switch between Opus, Sonnet, and Haiku via `/models` (interactive picker) or `/model <name>` (direct). Changing models restarts the session.

### Workspaces

Point Claude at any directory with `/workspace <path>`. Identity and memory from the home workspace carry over. Set a base directory with `/workspace base` to enable short names (e.g., `/workspace kai` instead of the full path). Create new workspaces with `/workspace new <name>`.

### Image and file support

Send photos or documents directly in chat. Supports images (JPEG, PNG, GIF, WebP) and text files (Python, JS, JSON, Markdown, and many more).

### Voice input

Voice notes are transcribed locally using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and forwarded to Claude. Requires `ffmpeg` and `whisper-cpp`. Disabled by default — set `VOICE_ENABLED=true` after installing dependencies. See the [Voice Setup](https://github.com/dcellison/kai/wiki/Voice-Setup) wiki page.

### Voice responses (TTS)

Text-to-speech via [Piper TTS](https://github.com/rhasspy/piper). Three modes: `/voice only` (voice note, no text), `/voice on` (text + voice), `/voice off` (text only, default). Eight curated English voices. Requires `pip install -e '.[tts]'` and `TTS_ENABLED=true`. See [Voice Setup](https://github.com/dcellison/kai/wiki/Voice-Setup).

### Webhooks

An HTTP server receives GitHub webhook events (pushes, PRs, issues, comments, reviews) and forwards them to Telegram. Signatures are validated via HMAC-SHA256. A generic webhook endpoint (`POST /webhook`) accepts JSON from any service. See [Exposing Kai to the Internet](https://github.com/dcellison/kai/wiki/Exposing-Kai-to-the-Internet).

### Scheduled jobs

Reminders and recurring Claude jobs with one-shot, daily, and interval schedules. Ask naturally ("remind me at 3pm") or use the HTTP API (`POST /api/schedule`). Claude jobs support conditional auto-remove for monitoring use cases (`CONDITION_MET` / `CONDITION_NOT_MET` protocol). See [Scheduling and Conditional Jobs](https://github.com/dcellison/kai/wiki/Scheduling-and-Conditional-Jobs).

### Memory

Three layers of persistent context:

1. **Auto-memory** — managed by Claude Code per-workspace. Project architecture and patterns.
2. **Home memory** (`workspace/.claude/MEMORY.md`) — personal memory, always injected regardless of current workspace. Proactively updated by Kai.
3. **Conversation history** (`workspace/.claude/history/`) — JSONL logs, one file per day. Searchable for past conversations.

Foreign workspaces also get their own `.claude/MEMORY.md` injected alongside home memory. See [Architecture](https://github.com/dcellison/kai/wiki/Architecture).

### Crash recovery

If interrupted mid-response, Kai notifies you on restart and asks you to resend your last message.

## Project Structure

```
kai/
├── src/kai/              # Source package
│   ├── __init__.py       # Version
│   ├── __main__.py       # python -m kai entry point
│   ├── main.py           # Async startup and shutdown
│   ├── bot.py            # Telegram handlers, commands, message routing
│   ├── claude.py         # Persistent Claude Code subprocess management
│   ├── config.py         # Environment config loading
│   ├── sessions.py       # SQLite session, job, and settings storage
│   ├── cron.py           # Scheduled job execution (APScheduler)
│   ├── webhook.py        # HTTP server: GitHub/generic webhooks, scheduling API
│   ├── history.py        # Conversation history (read/write JSONL logs)
│   ├── locks.py          # Per-chat async locks and stop events
│   ├── transcribe.py     # Voice message transcription (ffmpeg + whisper-cpp)
│   └── tts.py            # Text-to-speech synthesis (Piper TTS + ffmpeg)
├── tests/                # Test suite
├── models/               # Whisper and Piper model files (gitignored)
├── workspace/            # Claude Code working directory
│   └── .claude/          # Identity, memory, and chat history
├── pyproject.toml        # Package metadata and dependencies
├── Makefile              # Common dev commands
├── .env.example          # Environment variable template
└── LICENSE               # Apache 2.0
```

## Development

```bash
make install    # Install in editable mode with dev tools
make lint       # Run ruff linter
make format     # Auto-format with ruff
make check      # Lint + format check (CI-friendly)
make test       # Run test suite
make run        # Start the bot
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
