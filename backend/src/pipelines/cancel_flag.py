"""
Thread-safe shared flag to cooperatively cancel a running video analysis.

Set by the /analyze/video/cancel endpoint, checked periodically by
orchestrator.process_video() between frame batches. The analysis loop
is not interrupted immediately — it only notices the flag at the start
of its next batch iteration.
"""

import threading

_lock = threading.Lock()
_cancel_requested = False


def clear_cancel() -> None:
    """Reset the flag before starting a new analysis."""
    global _cancel_requested
    with _lock:
        _cancel_requested = False


def request_cancel() -> None:
    """Set the flag to request cancellation of the current analysis."""
    global _cancel_requested
    with _lock:
        _cancel_requested = True


def is_cancel_requested() -> bool:
    """Return True if the cancel flag is set."""
    global _cancel_requested
    with _lock:
        return _cancel_requested
