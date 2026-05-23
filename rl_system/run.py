"""
Compatibility wrapper for the canonical live runner.

Historically this file carried a full copy of the orchestration loop. The
implementation now lives in run_live.py so safety fixes and broker behavior do
not drift between two near-identical runners.
"""

from run_live import main


if __name__ == "__main__":
    main()
