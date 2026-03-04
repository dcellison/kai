"""Allow `python -m kai` to start the bot, or dispatch CLI subcommands."""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "totp":
    from kai.totp import cli

    cli(sys.argv[2:])
elif len(sys.argv) > 1 and sys.argv[1] == "install":
    from kai.install import cli as install_cli

    install_cli(sys.argv[2:])
else:
    from kai.main import main

    main()
