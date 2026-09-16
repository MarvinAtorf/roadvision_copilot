"""
Simple process-wide lock ensuring at most one video analysis runs at a time.

For MVP2, RoadVision Copilot supports exactly one analysis in flight across
the whole backend process. This module provides an atomic check-and-set so
concurrent /analyze/video requests (e.g. from multiple browser tabs) cannot
start a second analysis while one is already running.
"""

import threading

_lock = threading.Lock()
_in_progress = False


def try_start() -> bool:
    """
    Atomically check whether an analysis is running and, if not, claim the slot.

    :return: True if the caller successfully acquired the analysis slot and
        may proceed; False if another analysis is already in progress.
    """
    global _in_progress
    with _lock:
        if _in_progress:
            return False
        _in_progress = True
        return True


def finish() -> None:
    """
    Release the analysis slot.

    Must be called exactly once per successful `try_start()`, in a `finally`
    block, so the slot is always freed even if the analysis errors or is
    cancelled.
    """
    global _in_progress
    with _lock:
        _in_progress = False


def is_in_progress() -> bool:
    """Return whether an analysis is currently running, without claiming the slot."""
    with _lock:
        return _in_progress
