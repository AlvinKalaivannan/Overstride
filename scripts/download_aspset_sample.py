"""Download a validation slice of ASPset-510 for Section 4.

Fetches the small metadata archives (splits, cameras, boxes, joints_3d --
a few MB total) plus the ~1.4GB test-videos archive, extracting video for
just one subject to keep disk usage down (the archive itself bundles both
test-split subjects together, so the 1.4GB download is unavoidable, but we
don't keep the other subject's frames on disk).

    python scripts/download_aspset_sample.py                 # lists test subjects
    python scripts/download_aspset_sample.py --subject-id 1e28

Test split has exactly two subjects: 1e28 and 8a59 (confirmed from splits.csv).
"""

from __future__ import annotations

import argparse
import csv
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://archive.org/download/aspset510/"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "aspset510"
ARCHIVE_PREFIX = "ASPset-510"

METADATA_ARCHIVES = [
    "aspset510_v1_common-splits.tar.gz",
    "aspset510_v1_test-cameras.tar.gz",
    "aspset510_v1_test-boxes.tar.gz",
    "aspset510_v1_test-joints_3d.tar.gz",
]
VIDEOS_ARCHIVE = "aspset510_v1_test-videos.tar.gz"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  done ({dest.stat().st_size:,} bytes)")


def extract_all(archive_path: Path, data_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isreg():
                continue
            member.name = str(Path(member.name).relative_to(ARCHIVE_PREFIX))
            tar.extract(member, data_dir, filter="data")


def extract_subject_videos(archive_path: Path, data_dir: Path, subject_id: str) -> int:
    """Stream through the large videos archive, only writing one subject's files."""
    target_dir = f"{ARCHIVE_PREFIX}/test/videos/{subject_id}/"
    count = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isreg() or not member.name.startswith(target_dir):
                continue
            member.name = str(Path(member.name).relative_to(ARCHIVE_PREFIX))
            tar.extract(member, data_dir, filter="data")
            count += 1
    return count


def list_test_subjects_and_clips(data_dir: Path) -> list[tuple[str, str, str]]:
    splits_path = data_dir / "splits.csv"
    rows = []
    with splits_path.open(newline="") as f:
        for subject_id, clip_id, split, camera_id in csv.reader(f):
            if split == "test":
                rows.append((subject_id, clip_id, camera_id))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--archive-dir", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    archive_dir = args.archive_dir or (data_dir / "archives")
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading metadata archives (small, a few MB total)...")
    for name in METADATA_ARCHIVES:
        dest = archive_dir / name
        download(BASE_URL + name, dest)
        extract_all(dest, data_dir)

    rows = list_test_subjects_and_clips(data_dir)
    subjects = sorted({s for s, _, _ in rows})
    print(f"\nTest split has {len(subjects)} subjects, {len(rows)} clips.")

    if args.subject_id is None:
        print("Pass --subject-id to download video for one of these subjects:")
        for s in subjects:
            clip_count = sum(1 for subj, _, _ in rows if subj == s)
            print(f"  {s}  ({clip_count} clips)")
        return

    if args.subject_id not in subjects:
        raise SystemExit(f"{args.subject_id!r} is not a test-split subject. Choices: {subjects}")

    print(f"\nDownloading test-videos archive (~1.4GB, one-time -- bundles both test subjects)...")
    videos_dest = archive_dir / VIDEOS_ARCHIVE
    download(BASE_URL + VIDEOS_ARCHIVE, videos_dest)

    print(f"Extracting only subject {args.subject_id!r}'s videos...")
    n = extract_subject_videos(videos_dest, data_dir, args.subject_id)
    print(f"Extracted {n} video files to {data_dir / 'test' / 'videos' / args.subject_id}")


if __name__ == "__main__":
    main()
