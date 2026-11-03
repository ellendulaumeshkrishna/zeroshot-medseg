"""Allows ``python -m zsmedseg <command>``."""

import sys

from .cli import main

sys.exit(main())
