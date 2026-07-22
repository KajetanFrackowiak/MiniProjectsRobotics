from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent

ASSETS_DIR = REPO_ROOT / "assets"

ASSETS_PEG_DIR = ASSETS_DIR / "peg_in_a_hole"
ASSETS_HYDRO_DIR = ASSETS_DIR / "hydro"
ASSETS_PEG_DIR = ASSETS_DIR / "peg_in_a_hole"
ASSETS_HYDRO_DIR = ASSETS_DIR / "hydro"

PEG_URDF_PATH = ASSETS_PEG_DIR / "peg.urdf"
IIWA_URDF_PATH = ASSETS_DIR / "planar_iiwa14_no_collision.urdf"

TABLE_SDF_PATH = ASSETS_HYDRO_DIR / "extra_heavy_duty_table_surface_only_collision.sdf"
HOLE_SDF_PATH = ASSETS_PEG_DIR / "hole_chamfered.sdf"


for path in [
    PEG_URDF_PATH,
    IIWA_URDF_PATH,
    TABLE_SDF_PATH,
    HOLE_SDF_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)