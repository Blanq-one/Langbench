import sys
from pathlib import Path

# bot/polyglotbot is not an installed package; tests import it via bot/ on path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
