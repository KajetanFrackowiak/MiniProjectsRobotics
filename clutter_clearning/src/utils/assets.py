from pathlib import Path

SRC_DIR = Path(__file__).parent.parent 
REPO_ROOT = SRC_DIR.parent

ASSETS_DIR = REPO_ROOT / "assets"

ASSETS_HYDRO_DIR = ASSETS_DIR / "hydro"

DMD_PATH = ASSETS_DIR / "clutter_planning.dmd.yaml"
SCENARIO_PATH = ASSETS_DIR / "clutter.scenarios.yaml"

