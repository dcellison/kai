"""Allow `python -m kai` to start the bot, or dispatch CLI subcommands."""

import sys

# Lazy imports keep the bot's heavy dependencies (telegram, aiohttp, etc.)
# out of the lightweight CLI subcommands (totp, install).
if len(sys.argv) > 1 and sys.argv[1] == "totp":
    # TOTP provisioning CLI (setup, status, reset)
    from kai.totp import cli

    cli(sys.argv[2:])
elif len(sys.argv) > 1 and sys.argv[1] == "install":
    # Protected installation CLI (config, apply, status)
    from kai.install import cli as install_cli

    install_cli(sys.argv[2:])
else:
    # Default: start the Telegram bot
    from kai.main import main

    main()
