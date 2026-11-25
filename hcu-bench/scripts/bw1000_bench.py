"""Compatibility entrypoint. New runs should use hcu_bench.py (schema v2)."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("hcu_bench.py")), run_name="__main__")