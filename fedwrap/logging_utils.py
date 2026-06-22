"""Per-run file logger and JSONL history writer."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunLogger:
    """Holds the logger and its handlers so they can be closed cleanly."""
    logger: logging.Logger
    file_handler: logging.Handler
    stream_handler: logging.Handler

    def close(self) -> None:
        for h in (self.file_handler, self.stream_handler):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            try:
                self.logger.removeHandler(h)
            except Exception:
                pass


def setup_run_logger(out_dir: Path, name: str = "fedwrap.run") -> RunLogger:
    """Create a logger that writes to both ``run.log`` and stderr."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return RunLogger(logger=logger, file_handler=fh, stream_handler=sh)
