import os
import sys
from datetime import datetime


class TeeStream:
    def __init__(self, primary_stream, file_stream):
        self.primary_stream = primary_stream
        self.file_stream = file_stream

    @property
    def encoding(self):
        return getattr(self.primary_stream, "encoding", "utf-8")

    def write(self, message):
        self.primary_stream.write(message)
        self.file_stream.write(message)
        return len(message)

    def flush(self):
        self.primary_stream.flush()
        self.file_stream.flush()

    def isatty(self):
        return self.primary_stream.isatty()

    def fileno(self):
        return self.primary_stream.fileno()


def stop_run_logging(state):
    if not state or state.get("closed", False):
        return

    try:
        sys.stdout = state["original_stdout"]
        sys.stderr = state["original_stderr"]
    except Exception:
        pass

    try:
        state["log_file"].flush()
        state["log_file"].close()
    except Exception:
        pass

    state["closed"] = True


def finalize_run_logging(state):
    if not state:
        return

    # If an exception is still propagating, keep tee streams active so the
    # interpreter-emitted traceback is captured in the log file too.
    if sys.exc_info()[0] is not None:
        try:
            state["log_file"].flush()
        except Exception:
            pass
        return

    stop_run_logging(state)


def start_run_logging(log_dir="outputs/logs", script_name="run"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    log_file = open(log_path, mode="a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    state = {
        "original_stdout": original_stdout,
        "original_stderr": original_stderr,
        "log_file": log_file,
        "closed": False,
        "log_path": log_path,
    }
    return state, log_path
