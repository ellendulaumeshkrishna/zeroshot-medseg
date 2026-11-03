"""Local demo website (``python -m zsmedseg serve``)."""

from .server import create_app, run_server

__all__ = ["create_app", "run_server"]
