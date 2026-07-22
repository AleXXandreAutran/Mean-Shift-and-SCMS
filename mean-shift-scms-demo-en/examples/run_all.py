"""Run every example."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for script_name in ("run_2d.py", "run_3d.py"):
        script = ROOT / "examples" / script_name
        print(f"\n--- {script_name} ---")
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
