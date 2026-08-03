from pathlib import Path

# Path(__file__) is the path to this file
# Path(__file__).parent to utils
# Path(__file__).parent.parent to src
SRC_DIR = Path(__file__).parent.parent
REPO_DIR = SRC_DIR.parent
ASSETS_DIR = REPO_DIR / "assets"

IIWA_DIR = ASSETS_DIR / "iiwa_description"
IIWA_SDF_7_NO_COLLISION = IIWA_DIR / "sdf" / "iiwa7_no_collision.sdf"
IIWA_SDF_7_WITH_BOX_COLLISION = IIWA_DIR / "sdf" / "iiwa7_with_box_collision.sdf"
YCB_DIR = ASSETS_DIR / "ycb"
SUGAR_BOX = YCB_DIR / "004_sugar_box.sdf"

SCHUNK_WSG_50_TIP = ASSETS_DIR / "schunk_wsg_50_with_tip.sdf"
AMAZON_TABLE_SIMPLIFIED = ASSETS_DIR / "amazon_table_simplified.sdf"
CUPBOARD = ASSETS_DIR / "cupboard.sdf"
CAMERA_BOX = ASSETS_DIR / "camera_box.sdf"


IIWA_SDF_7_NO_COLLISION_URI = f"file://{IIWA_SDF_7_NO_COLLISION}"
IIWA_SDF_7_WITH_BOX_COLLISION_URI = f"file://{IIWA_SDF_7_WITH_BOX_COLLISION}"
SUGAR_BOX_URI = f"file://{SUGAR_BOX}"
SCHUNK_WSG_50_TIP_URI = f"file://{SCHUNK_WSG_50_TIP}"
AMAZON_TABLE_SIMPLIFIED_URI = f"file://{AMAZON_TABLE_SIMPLIFIED}"
CUPBOARD_URI = f"file://{CUPBOARD}"
CAMERA_BOX_URI = f"file://{CAMERA_BOX}"
