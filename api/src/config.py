import os

# Absolute path to repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

SEQ_LEN = 5

PT_DIR = os.path.join(ROOT, "Pitch Type Prediction", "artifacts") + os.sep
LOC_DIR = os.path.join(ROOT, "Pitch Location Prediction", "artifacts") + os.sep
SHARED_DIR = os.path.join(ROOT, "artifacts", "shared") + os.sep
SERVING_TABLE_PATH = os.path.join(ROOT, "artifacts", "serving", "serving_table.parquet")