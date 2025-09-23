# extract test indices used

import os
import sys
import argparse
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-p",
    "--p",
    type=int,
    default=20,
    help="The p indicates what *percent* you want to dedicate out of the full data for calibration."
    "Default is 20%, in which case p=20.",
)
args = parser.parse_args()

params = {
    "data_src": args.data_name,
    "random_seed": 1952,
}

calib_size = 0.01 * args.p
# Open dataset
outputs = du.open_mass_dataset(
    name=params["data_src"],
    data_dir=Path(__file__).parent.parent.parent / "data",
    test_size=1 - calib_size,
    random_state=params["random_seed"],
)
print(*outputs["idx_test"])
