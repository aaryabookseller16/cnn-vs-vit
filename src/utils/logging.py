# src/utils/logging.py
from pathlib import Path
import csv

class CSVLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._fh = None

    def log(self, row: dict):
        if self._fh is None:
            self._fh = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None