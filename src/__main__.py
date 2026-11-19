"""Allow ``python -m src`` as a shortcut for ``python -m src.main``."""

import sys

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
