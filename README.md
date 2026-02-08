# Kai

A personal AI assistant accessed via Telegram, powered by [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Kai acts as a Telegram gateway to a persistent Claude Code CLI process. Messages you send in Telegram are forwarded to Claude, and responses stream back in real time. Claude has full tool access (shell, files, web search) and maintains conversation context across messages.

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
| `CLAUDE_MAX_BUDGET_USD` | No | `1.0` | Max spend per session |
| `WEBHOOK_PORT` | No | `8080` | HTTP server port for webhooks and scheduling API |
| `WEBHOOK_SECRET` | No | | Secret for webhook validation and scheduling API auth |

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
| `/workspace` | Show current working directory |
| `/workspace <path>` | Switch to a different repo/directory |
| `/workspace <number>` | Switch by history number |
| `/workspace home` | Return to default workspace |
| `/workspaces` | List recently used workspaces |
| `/stop` | Interrupt a response mid-stream |
| `/memory` | View persistent memory |
| `/memory clear` | Clear all memory |
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
Point Kai at any directory on your machine with `/workspace <path>`. Kai's identity and memory carry over from the home workspace. Use `/workspaces` to see a numbered list of recent workspaces and switch quickly with `/workspace <number>`.

### Image and file support
Send photos or documents directly in the chat. Kai supports:
- **Images** (JPEG, PNG, GIF, WebP) - sent as photos or uncompressed documents
- **Text files** (Python, JS, JSON, Markdown, and many more) - content is extracted and sent to Claude

### GitHub notifications

Kai runs an HTTP server that receives GitHub webhook events and forwards them to Telegram as formatted notifications. Supported events:

- **Pushes** — commit summaries with SHAs, messages, and a compare link
- **Pull requests** — opened, closed, merged, reopened
- **Issues** — opened, closed, reopened
- **Issue comments** — new comments with author and body preview
- **PR reviews** — approved or changes requested

Signatures are validated using the `WEBHOOK_SECRET` via HMAC-SHA256, matching GitHub's `X-Hub-Signature-256` header.

To configure, add a webhook on your GitHub repo pointing to `https://<your-host>/webhook/github` with the secret set to your `WEBHOOK_SECRET` value. Select the events you want.

### Generic webhooks

POST JSON to `/webhook` with an `X-Webhook-Secret` header. If the payload contains a `message` field, that text is sent to Telegram; otherwise the full JSON is forwarded (truncated to 4096 characters).

Useful for connecting any service that can fire HTTP requests — CI pipelines, monitoring alerts, home automation, etc.

### Scheduled jobs

Kai can schedule reminders and recurring tasks. Ask it naturally (e.g., "remind me to check the laundry at 3pm") and it will use the built-in scheduling API. Two job types:

- **Reminders** — sends a message at the scheduled time
- **Claude jobs** — runs a prompt through Claude at the scheduled time (useful for monitoring, daily summaries, etc.)

Jobs support one-shot, daily, and interval schedules. Use `/jobs` to list active jobs and `/canceljob <id>` to remove one.

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

Kai remembers facts across sessions. Ask it to remember something and it will persist to `.claude/MEMORY.md` in the workspace. Memory survives `/new` and model switches.

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
│   └── locks.py          # Per-chat async locks and stop events
├── tests/                # Test suite
├── workspace/            # Claude Code working directory
│   └── .claude/          # Identity and memory
├── pyproject.toml        # Package metadata, dependencies, and tool config
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
