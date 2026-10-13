"""Allow ``python -m opcontinuity``."""

import sys

from opcontinuity.cli import main

if __name__ == "__main__":
    sys.exit(main())
