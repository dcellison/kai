# Kai

A personal AI assistant accessed via Telegram, powered by [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Kai acts as a Telegram gateway to a persistent Claude Code CLI process. Messages you send in Telegram are forwarded to Claude, and responses stream back in real time. Claude has full tool access (shell, files, web search) and maintains conversation context across messages.

### Why Kai?

**Security-first.** Projects like OpenClaw route your messages through third-party servers with unclear data handling. Kai runs entirely on your own machine — your code, your credentials, and your conversations never leave your hardware. The only external connections are to the Telegram Bot API and Anthropic's API, both authenticated and encrypted.

**Developer-focused.** Kai is built around Claude Code's full capabilities: shell access, file editing, web search, and tool use. Switch between repos from your phone, create workspaces, trigger builds, review diffs — all through Telegram. It's a remote development companion, not just a chatbot.

**Why Telegram?** Telegram's Bot API is the most capable messaging platform for this use case. It supports message editing (enabling real-time streaming output), inline keyboards (interactive UI for model/workspace switching), file and image handling, slash commands, and unlimited free messaging. No other major platform offers all of these without restrictions or per-message costs. See the [project wiki](https://github.com/dcellison/kai/wiki) for a detailed comparison of messaging platforms evaluated during development.

**Why Claude Code?** Claude Code provides a persistent CLI with full tool access — shell commands, file operations, web search — in a single subprocess. Kai doesn't need to implement its own tool-use layer or manage API conversations directly. It delegates to Claude Code and focuses on the Telegram interface, workspace management, and scheduling. When authenticated via `claude login` on a Max plan, all usage is covered by the subscription — no per-token API costs.

**Why local?** Kai is portable enough to run on a VPS, but local deployment is a deliberate choice. Running on your own machine enables flat-rate Max plan authentication (no per-token API costs), access to local applications and repos (macOS Calendar, Music, development tools), and a clear security boundary — your conversations and credentials never leave your hardware. A VPS would trade all three for always-on hosting, which launchd (macOS) or systemd (Linux) already provide on a local machine.

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
```

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | | Bot token from BotFather |
| `ALLOWED_USER_IDS` | Yes | | Comma-separated Telegram user IDs |
| `CLAUDE_MODEL` | No | `sonnet` | Default model (`opus`, `sonnet`, or `haiku`) |
| `CLAUDE_TIMEOUT_SECONDS` | No | `120` | Per-message timeout |
| `CLAUDE_MAX_BUDGET_USD` | No | `10.0` | Session budget cap (see below) |
| `WEBHOOK_PORT` | No | `8080` | HTTP server port for webhooks and scheduling API |
| `WEBHOOK_SECRET` | No | | Secret for webhook validation and scheduling API auth |
| `VOICE_ENABLED` | No | `false` | Enable voice message transcription (see below) |

### Session budget cap

`CLAUDE_MAX_BUDGET_USD` is passed to Claude Code's `--max-budget-usd` flag. It limits how much work the inner Claude can do in a single session, measured in estimated API token costs. When the cap is reached, Claude stops processing and you'll need to start a new session with `/new`.

On the Max plan (subscription-based), no per-token charges are incurred — the budget acts purely as a runaway prevention mechanism. On API-billed plans, it caps actual spend. The session resets whenever you use `/new`, switch models, or switch workspaces.

## Running

```bash
source .venv/bin/activate
python -m kai
```

Kai will start polling for Telegram messages. Press Ctrl+C to stop.

### Running as a service (macOS)

Create a launchd plist to keep Kai running in the background and restart it automatically. Replace `/Users/kai/Projects/kai` with your actual install path:

```bash
cat > ~/Library/LaunchAgents/com.kai.bot.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kai.bot</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/kai/Projects/kai/.venv/bin/python</string>
        <string>-m</string>
        <string>kai</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/kai/Projects/kai</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/kai/Projects/kai/kai.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/kai/Projects/kai/kai.log</string>

    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
EOF
```

Load and manage the service:

```bash
# Start
launchctl load ~/Library/LaunchAgents/com.kai.bot.plist

# Stop
launchctl unload ~/Library/LaunchAgents/com.kai.bot.plist

# Restart (stop then start)
launchctl unload ~/Library/LaunchAgents/com.kai.bot.plist
launchctl load ~/Library/LaunchAgents/com.kai.bot.plist

# Check status
launchctl list | grep com.kai.bot
```

### Running as a service (Linux with systemd)

Adjust paths to match your install location:

```ini
# /etc/systemd/system/kai.service
[Unit]
Description=Kai Telegram Bot
After=network.target

[Service]
Type=simple
User=kai
WorkingDirectory=/home/kai/kai
ExecStart=/home/kai/kai/.venv/bin/python -m kai
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable kai
sudo systemctl start kai
sudo systemctl status kai
```

## Telegram Commands

| Command | Description |
|---|---|
| `/models` | Interactive model picker with inline buttons |
| `/model <name>` | Switch model directly (`opus`, `sonnet`, `haiku`) |
| `/new` | Clear session and start fresh |
| `/workspace` | Show current workspace |
| `/workspace <name>` | Switch by name (resolved via base) or absolute path |
| `/workspace home` | Return to default workspace |
| `/workspace base <path>` | Set the projects directory for short-name resolution |
| `/workspace base` | Show current base directory |
| `/workspace new <name>` | Create a new workspace, git init, and switch to it |
| `/workspaces` | Interactive workspace picker (inline buttons) |
| `/stop` | Interrupt a response mid-stream |
| `/memory` | Show memory file locations |
| `/stats` | Show session info, model, and cost |
| `/jobs` | List active scheduled jobs |
| `/canceljob <id>` | Cancel a scheduled job |
| `/webhooks` | Show webhook server status |
| `/help` | Show available commands |

## Features

### Streaming responses
Responses stream into Telegram in real time, updating the message every 2 seconds as Claude generates text.

### Model switching
Use `/models` for an interactive picker or `/model <name>` to switch directly. Changing models restarts the session.

### Workspace switching
Point Kai at any directory on your machine with `/workspace <path>`. Kai's identity and memory carry over from the home workspace.

To switch by short name (e.g., `/workspace kai` instead of `/workspace /Users/you/Projects/kai`), first set a base directory:

```
/workspace base /Users/you/Projects
```

After that, `/workspace kai` resolves to `/Users/you/Projects/kai`. Without a base, only absolute paths and `~` paths work.

Create new workspaces with `/workspace new <name>` (creates the directory and runs `git init`). Use `/workspaces` for an interactive picker with inline buttons — tap to switch, or tap the current workspace to dismiss.

### Image and file support
Send photos or documents directly in the chat. Kai supports:
- **Images** (JPEG, PNG, GIF, WebP) - sent as photos or uncompressed documents
- **Text files** (Python, JS, JSON, Markdown, and many more) - content is extracted and sent to Claude

### Voice messages

Send a voice note in Telegram and Kai transcribes it locally using [whisper.cpp](https://github.com/ggerganov/whisper.cpp), then forwards the transcription to Claude. The transcription is echoed back to the chat so you can see what Kai heard before it responds.

Everything runs on your machine — no external speech-to-text APIs or per-minute costs. Requires `ffmpeg` and `whisper-cpp` (both available via Homebrew) plus a one-time model download (~148MB).

Voice messages are disabled by default. Set `VOICE_ENABLED=true` in `.env` after installing the dependencies. See the [Voice Message Setup](https://github.com/dcellison/kai/wiki/Voice-Message-Setup) wiki page for full instructions.

### GitHub notifications

Kai runs an HTTP server that receives GitHub webhook events and forwards them to Telegram as formatted notifications. Supported events:

- **Pushes** — commit summaries with SHAs, messages, and a compare link
- **Pull requests** — opened, closed, merged, reopened
- **Issues** — opened, closed, reopened
- **Issue comments** — new comments with author and body preview
- **PR reviews** — approved or changes requested

Signatures are validated using the `WEBHOOK_SECRET` via HMAC-SHA256, matching GitHub's `X-Hub-Signature-256` header.

The webhook server listens on `localhost:8080` by default. To receive events from GitHub (or any external service), you need to expose it to the internet. Options include:

- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) — free, runs as a background service, maps a domain to your local port. Recommended if you have a domain on Cloudflare.
- [ngrok](https://ngrok.com/) — quick setup, gives you a public URL with one command. Good for testing.
- A reverse proxy (nginx, Caddy) on a server with a public IP.

If using Cloudflare Tunnel, the tunnel config should only route public-facing paths to your server. Internal APIs (like `/api/*`) should not be exposed. A config like this ensures only webhooks and health checks are reachable from the internet — everything else returns 404 at the tunnel level, before it ever reaches your server:

```yaml
ingress:
  - hostname: your.domain
    path: /webhook/*
    service: http://localhost:8080
  - hostname: your.domain
    path: /health
    service: http://localhost:8080
  - service: http_status:404
```

Once exposed, add a webhook on your GitHub repo pointing to `https://<your-host>/webhook/github` with the secret set to your `WEBHOOK_SECRET` value. Select the events you want. See the [Exposing Kai to the Internet](https://github.com/dcellison/kai/wiki/Exposing-Kai-to-the-Internet) wiki page for a full walkthrough.

### Generic webhooks

POST JSON to `/webhook` with an `X-Webhook-Secret` header. If the payload contains a `message` field, that text is sent to Telegram; otherwise the full JSON is forwarded (truncated to 4096 characters).

Useful for connecting any service that can fire HTTP requests — CI pipelines, monitoring alerts, home automation, etc.

### Scheduled jobs

Kai can schedule reminders and recurring tasks. Ask it naturally (e.g., "remind me to check the laundry at 3pm") and it will use the built-in scheduling API. Two job types:

- **Reminders** — sends a message at the scheduled time
- **Claude jobs** — runs a prompt through Claude at the scheduled time (useful for monitoring, daily summaries, etc.)

Jobs support one-shot, daily, and interval schedules. Use `/jobs` to list active jobs and `/canceljob <id>` to remove one. See the [Scheduling and Conditional Jobs](https://github.com/dcellison/kai/wiki/Scheduling-and-Conditional-Jobs) wiki page for details on job types, the HTTP API, and conditional monitoring patterns.

#### Conditional jobs

Claude jobs can be set with `auto_remove: true` for monitoring use cases. Claude is expected to respond with a protocol marker:

- `CONDITION_MET: <message>` — delivers the message and deactivates the job
- `CONDITION_NOT_MET` — silently continues checking on the next scheduled run

This is useful for things like "let me know when this PR is merged" or "tell me when the deploy finishes."

#### Scheduling HTTP API

Jobs can also be created programmatically via `POST /api/schedule`:

```json
{
  "name": "daily standup",
  "prompt": "Summarize my recent git activity",
  "schedule_type": "daily",
  "schedule_data": {"time": "09:00"},
  "job_type": "claude",
  "auto_remove": false
}
```

Auth: set the `X-Webhook-Secret` header to your `WEBHOOK_SECRET`. Schedule types: `once` (with `{"run_at": "ISO8601"}`), `daily` (with `{"time": "HH:MM"}`), `interval` (with `{"seconds": N}`).

### Persistent memory

Kai has two layers of memory, injected at the start of each session:

1. **Auto-memory** (`~/.claude/projects/.../memory/MEMORY.md`) — managed automatically by Claude Code. Project architecture, completed features, infrastructure knowledge. Created per-workspace.
2. **Home memory** (`workspace/.claude/MEMORY.md`) — Kai's personal memory from the home workspace. User preferences, facts, ongoing context. Always injected regardless of current workspace.

When working in a foreign workspace, Kai also injects that workspace's `.claude/MEMORY.md` if it exists, so project-specific context is available alongside personal memory.

Auto-memory is institutional knowledge (how the project works) while home memory is personal (who you are, what you prefer). By injecting both, Kai always has its full context regardless of which workspace it's in.

Use `/memory` to see file locations. The [Architecture](https://github.com/dcellison/kai/wiki/Architecture) wiki page covers how memory injection works across workspace switches.

### Chat logging

All messages are logged as JSONL files in `workspace/chat_history/`, organized by date.

### Crash recovery

If Kai is interrupted mid-response, it notifies you on restart and asks you to resend your last message.

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
│   ├── chat_log.py       # JSONL chat logging
│   ├── locks.py          # Per-chat async locks and stop events
│   └── transcribe.py     # Voice message transcription (ffmpeg + whisper-cpp)
├── tests/                # Test suite
├── models/               # Whisper model files (gitignored)
├── workspace/            # Claude Code working directory
│   └── .claude/          # Identity (CLAUDE.md) and memory template (MEMORY.md.example)
├── pyproject.toml        # Package metadata, dependencies, and tool config
├── Makefile              # Common dev commands
├── .env.example          # Environment variable template
└── LICENSE               # Apache 2.0
```

For a deeper look at the message lifecycle, database schema, concurrency model, and how workspaces interact with memory, see the [Architecture](https://github.com/dcellison/kai/wiki/Architecture) wiki page.

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
