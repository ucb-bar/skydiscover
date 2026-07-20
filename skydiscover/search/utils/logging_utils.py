"""Logging utilities for the search subsystem.

Provides a compact console formatter, a noise-reducing filter, and a
one-call helper to wire up both file and console logging for a
discovery run.
"""

import logging
import os
import time

_QUIET = {"route", "server"}


class _ConsoleFormatter(logging.Formatter):
    """Compact single-line formatter: ``HH:MM:SS [module] message``."""

    def format(self, record):
        ts = self.formatTime(record, "%H:%M:%S")
        name = (
            record.name[len("skydiscover.") :]
            if record.name.startswith("skydiscover.")
            else record.name
        )
        parts = name.split(".")
        short = f"search.{parts[1]}" if parts[0] == "search" and len(parts) >= 3 else parts[-1]
        fmt = (
            f"{ts} {record.levelname} [{short}] "
            if record.levelno >= logging.WARNING
            else f"{ts} [{short}] "
        )
        return fmt + record.getMessage()


class _ConsoleFilter(logging.Filter):
    """Only pass skydiscover messages, suppressing noisy modules below WARNING."""

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        if not record.name.startswith("skydiscover") or record.name.split(".")[-1] in _QUIET:
            return False
        return True


def setup_search_logging(log_level: str, log_dir: str, name: str) -> None:
    """Configure logging with a timestamped file handler and a console handler.

    Handlers are placed on the ``skydiscover`` named logger with
    ``propagate=False`` so that Ray's root-level StreamHandler (or any
    other framework handler on root) does not duplicate console output.
    """
    os.makedirs(log_dir, exist_ok=True)
    level = getattr(logging, log_level)

    log_file = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Root: file-only (no StreamHandler). Non-skydiscover messages still
    # reach the log file; _ConsoleFilter already suppressed them on console.
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    root.setLevel(level)
    fh_root = logging.FileHandler(log_file)
    fh_root.setFormatter(fmt)
    root.addHandler(fh_root)

    # 'skydiscover' logger: file + console, propagate=False.
    sky = logging.getLogger("skydiscover")
    for h in sky.handlers[:]:
        sky.removeHandler(h)
        h.close()
    sky.setLevel(level)
    sky.propagate = False

    fh_sky = logging.FileHandler(log_file)
    fh_sky.setFormatter(fmt)
    sky.addHandler(fh_sky)

    ch = logging.StreamHandler()
    ch.setFormatter(_ConsoleFormatter())
    ch.addFilter(_ConsoleFilter())
    sky.addHandler(ch)

    # 'evolve_flows' logger: same pattern as 'skydiscover' — file + console,
    # propagate=False.  Prevents duplication when Ray (or another framework)
    # reinstates a StreamHandler on root after we strip it above.
    ef = logging.getLogger("evolve_flows")
    for h in ef.handlers[:]:
        ef.removeHandler(h)
        h.close()
    ef.setLevel(level)
    ef.propagate = False

    fh_ef = logging.FileHandler(log_file)
    fh_ef.setFormatter(fmt)
    ef.addHandler(fh_ef)

    ch_ef = logging.StreamHandler()
    ch_ef.setFormatter(_ConsoleFormatter())
    ef.addHandler(ch_ef)

    logging.getLogger(__name__).info(f"Logging to {log_file}")
