"""Run the local PAI adapter from a repository checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stream4d.pai.convert import main

if __name__ == "__main__":
    main()
