import sys
from typing import Any, Union, Optional

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
