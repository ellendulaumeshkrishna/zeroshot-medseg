"""Logging setup shared by the command-line tools."""

from __future__ import annotations

import logging
import sys
import warnings


def setup_logging(level: int = logging.INFO) -> None:
    """Log this package at ``level``; keep third-party libraries quiet.

    SAM2 logs every image embedding at INFO on the root logger, so the root logger is set
    to WARNING and only the ``zsmedseg`` logger is raised to ``level``.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("zsmedseg").setLevel(level)
    # torchvision / timm / sam2 emit deprecation and performance warnings on every call.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
