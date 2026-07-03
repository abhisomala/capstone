import sys
from pathlib import Path

# Ensure `import src...` resolves when pytest is run from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent))
