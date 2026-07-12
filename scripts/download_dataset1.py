"""Download Dataset #1 (Mid-Long Distance Runners Injuries, weekly timeseries)
into data/raw/. Public GitHub raw file, no credentials required.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/josedv82/public_sport_science_datasets/"
    "main/Mid-Long%20Distance%20Runners%20Injuries/timeseries%20(weekly).csv"
)

DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "mid_long_distance_runners_injuries" / "timeseries_weekly.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Done ({DEST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
