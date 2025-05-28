from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config
DETECTION_MODEL_DIR = ROOT
Model_dictionary = {"cat_sick_yolo11n": DETECTION_MODEL_DIR / "cat_sick_yolo11n.pt",
                    "cat_sick_yolo11m": DETECTION_MODEL_DIR / "cat_sick_yolo11m.pt",
                    "cat_sick_yolo11l": DETECTION_MODEL_DIR / "cat_sick_yolo11l.pt",
                    }


DETECTION_MODEL_LIST = [
    "cat_sick_yolo11n",
"cat_sick_yolo11m",
"cat_sick_yolo11l",
]