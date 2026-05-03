"""Allow `python -m kmx ...` to invoke the CLI entry point."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
