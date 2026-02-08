# Kai

A personal AI assistant accessed via Telegram, powered by [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Kai acts as a Telegram gateway to a persistent Claude Code CLI process. Messages you send in Telegram are forwarded to Claude, and responses stream back in real time. Claude has full tool access (shell, files, web search) and maintains conversation context across messages.

## Requirements

- Python 3.9+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- Your Telegram user ID (get it from [@userinfobot](https://t.me/userinfobot))

## Setup

```bash
git clone git@github.com:dcellison/kai.git
cd kai
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
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

## Running

```bash
source .venv/bin/activate
python main.py
```

Kai will start polling for Telegram messages. Press Ctrl+C to stop.

### Running as a service (macOS)

Create a launchd plist to keep Kai running in the background and restart it automatically:

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
        <string>/Users/kai/Projects/kai/main.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/kai/Projects/kai</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/kai/Projects/kai/kai.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/kai/Projects/kai/kai.log</string>
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

```ini
# /etc/systemd/system/kai.service
[Unit]
Description=Kai Telegram Bot
After=network.target

[Service]
Type=simple
User=kai
WorkingDirectory=/home/kai/kai
ExecStart=/home/kai/kai/.venv/bin/python main.py
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
| `/stop` | Interrupt a response mid-stream |
| `/stats` | Show session info, model, and cost |
| `/jobs` | List active scheduled jobs |
| `/canceljob <id>` | Cancel a scheduled job |
| `/help` | Show available commands |

## Features

### Streaming responses
Responses stream into Telegram in real time, updating the message every 2 seconds as Claude generates text.

### Model switching
Use `/models` for an interactive picker or `/model <name>` to switch directly. Changing models restarts the session.

### Image and file support
Send photos or documents directly in the chat. Kai supports:
- **Images** (JPEG, PNG, GIF, WebP) - sent as photos or uncompressed documents
- **Text files** (Python, JS, JSON, Markdown, and many more) - content is extracted and sent to Claude

### Scheduled jobs
Kai can schedule reminders and recurring tasks. Ask it naturally (e.g., "remind me to check the laundry at 3pm") and it will use the built-in scheduling system. Two job types:

- **Reminders** - sends a message at the scheduled time
- **Claude jobs** - runs a prompt through Claude at the scheduled time (useful for monitoring, daily summaries, etc.)

Jobs support one-shot, daily, and interval schedules. Use `/jobs` to list active jobs and `/canceljob <id>` to remove one.

### Chat logging
All messages are logged as JSONL files in `workspace/chat_history/`, organized by date.

### Crash recovery
If Kai is interrupted mid-response, it notifies you on restart and asks you to resend your last message.

## Project Structure

```
kai/
├── main.py           # Entry point, async startup and shutdown
├── bot.py            # Telegram handlers, commands, message routing
├── claude.py         # Persistent Claude Code subprocess management
├── config.py         # Environment config loading
├── sessions.py       # SQLite session and job storage
├── cron.py           # Scheduled job execution (APScheduler)
├── chat_log.py       # JSONL chat logging
├── locks.py          # Per-chat async locks and stop events
├── workspace/        # Claude Code working directory
│   └── schedule_job.py   # CLI helper for creating scheduled jobs
├── .env.example      # Environment variable template
└── pyproject.toml    # Project metadata and dependencies
```

## License

Private project. Not licensed for redistribution.
