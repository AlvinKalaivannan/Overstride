"""Score one athlete-week against their incrementally-updated baseline.

    python scripts/score_week.py --athlete-id 12 --features week.json

`week.json` maps each of the frozen model's feature column names to a
value, e.g. {"nr. sessions": 5.0, "total kms": 22.2, ...}.

Per-athlete baseline state persists between runs as JSON under
--state-dir (default data/interim/baselines/, gitignored) — this is a
minimal file-based store for wiring the pipeline together; a real
deployment would swap it for a database without touching score.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from overstride.baseline.welford import WelfordBaseline
from overstride.risk.score import FrozenModel, process_week

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "stage2_logistic_coeffs.json"
DEFAULT_STATE_DIR = ROOT / "data" / "interim" / "baselines"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athlete-id", required=True)
    parser.add_argument("--features", required=True, type=Path, help="JSON file of feature name -> value")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    args = parser.parse_args()

    model = FrozenModel.load(args.model)
    feature_values = json.loads(args.features.read_text(encoding="utf-8"))
    vector = [feature_values[name] for name in model.feature_columns]

    state_path = args.state_dir / f"{args.athlete_id}.json"
    if state_path.exists():
        baseline = WelfordBaseline.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
    else:
        baseline = WelfordBaseline()

    result = process_week(baseline, vector, model)

    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(baseline.to_dict()), encoding="utf-8")

    print(json.dumps(
        {
            "status": result.status,
            "clean_weeks": result.clean_weeks,
            "d2": result.d2,
            "probability": result.probability,
            "feature_contributions": result.feature_contributions,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
