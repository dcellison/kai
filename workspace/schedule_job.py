#!/usr/bin/env python3
"""CLI helper for the inner Claude to schedule jobs directly in the database.

Usage examples:
    # One-shot reminder
    python schedule_job.py --name "Laundry" --prompt "Do the laundry!" \
        --schedule-type once --run-at "2026-02-08T14:00:00+00:00"

    # Daily reminder at 14:00 UTC
    python schedule_job.py --name "Standup" --prompt "Time for standup" \
        --schedule-type daily --time "14:00"

    # Repeating every 3600 seconds
    python schedule_job.py --name "Check mail" --prompt "Check your email" \
        --schedule-type interval --seconds 3600

    # Claude-type job (processed by Claude, not just a message)
    python schedule_job.py --name "Weather" --job-type claude \
        --prompt "What's the weather today?" --schedule-type daily --time "08:00"

    # Auto-remove job (deactivates when condition is met)
    python schedule_job.py --name "Package tracker" --job-type claude --auto-remove \
        --prompt "Has my package arrived?" --schedule-type interval --seconds 3600
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "sessions.db"
SIGNAL_DIR = Path(__file__).resolve().parent / ".cron"
SIGNAL_FILE = SIGNAL_DIR / ".pending"


def main() -> None:
    parser = argparse.ArgumentParser(description="Schedule a job for the Kai bot")
    parser.add_argument("--name", required=True, help="Job name")
    parser.add_argument("--prompt", required=True, help="Message or prompt text")
    parser.add_argument("--job-type", default="reminder", choices=["reminder", "claude"],
                        help="Job type (default: reminder)")
    parser.add_argument("--schedule-type", required=True, choices=["once", "daily", "interval"],
                        help="Schedule type")
    parser.add_argument("--auto-remove", action="store_true",
                        help="Auto-remove when condition is met (claude jobs only)")

    # Schedule-type-specific args
    parser.add_argument("--run-at", help="ISO datetime for one-shot jobs")
    parser.add_argument("--time", help="HH:MM (UTC) for daily jobs")
    parser.add_argument("--seconds", type=int, help="Interval in seconds for repeating jobs")

    # Chat ID (defaults to reading from DB — there's typically only one user)
    parser.add_argument("--chat-id", type=int, help="Telegram chat ID (auto-detected if omitted)")

    args = parser.parse_args()

    # Build schedule_data
    schedule_type = args.schedule_type
    if schedule_type == "once":
        if not args.run_at:
            parser.error("--run-at is required for schedule-type 'once'")
        schedule_data = json.dumps({"run_at": args.run_at})
    elif schedule_type == "daily":
        if not args.time:
            parser.error("--time is required for schedule-type 'daily'")
        schedule_data = json.dumps({"time": args.time})
    elif schedule_type == "interval":
        if not args.seconds:
            parser.error("--seconds is required for schedule-type 'interval'")
        schedule_data = json.dumps({"seconds": args.seconds})
    else:
        parser.error(f"Unknown schedule type: {schedule_type}")

    if not DB_PATH.exists():
        print(f"Error: database not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Auto-detect chat_id if not provided
        chat_id = args.chat_id
        if chat_id is None:
            row = conn.execute("SELECT chat_id FROM sessions LIMIT 1").fetchone()
            if row is None:
                print("Error: no sessions in DB. Provide --chat-id explicitly.", file=sys.stderr)
                sys.exit(1)
            chat_id = row[0]

        cursor = conn.execute(
            """INSERT INTO jobs (chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, args.name, args.job_type, args.prompt, schedule_type, schedule_data,
             int(args.auto_remove)),
        )
        conn.commit()
        job_id = cursor.lastrowid
    finally:
        conn.close()

    # Signal the bot to pick up the new job
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_FILE.touch()

    print(f"Job #{job_id} '{args.name}' scheduled ({schedule_type})")


if __name__ == "__main__":
    main()
