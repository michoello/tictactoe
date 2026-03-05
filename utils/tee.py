from typing import Any, Union, Optional
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import builtins

prev_ts = -1
_TS_RE = re.compile(r"\[ts:(\d+)\]")

def timestamped_print(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> int:
    ts = int(time.time())

    global prev_ts
    if prev_ts == -1:
        prev_ts = ts

    timestamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime("%Y-%m-%d %H:%M:%S")
    message = sep.join(str(arg) for arg in args)
    m = _TS_RE.search(message)
    if m:
        prev_ts = int(m.group(1))
        message = _TS_RE.sub("", message, count=1).strip()

    dif_ts = ts - prev_ts
    if dif_ts > 1:
        prev_ts = ts

    lines = message.splitlines()
    timestamped_lines = [f"{dif_ts:5d} - {timestamp} - {line}" for line in lines]
    final_message = "\n".join(timestamped_lines)
    builtins.print(final_message, end=end, file=file, flush=flush)
    return ts

import sys

class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: Any) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

def setup_logging(log_file_path: Union[dict[str, Any], str, None] = "output.log") -> None:
    if log_file_path:
        # Assuming the argument passed might be dict if parsed awkwardly, standardizing to str
        if isinstance(log_file_path, dict):
            path = log_file_path.get("log_file_path", "output.log") 
        else:
            path = log_file_path
        
        logfile = open(path, "w")
        sys.stdout = Tee(sys.stdout, logfile)  # type: ignore
