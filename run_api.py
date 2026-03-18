"""Launcher script to run the API with correct paths."""
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Now import and run
from api.main_extended import app
import uvicorn

if __name__ == "__main__":
    print("Starting LoanLens API...")
    print(f"Python path: {sys.path[:2]}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
