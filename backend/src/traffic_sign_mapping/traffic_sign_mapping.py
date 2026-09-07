import json
import os
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "traffic_sign_mapping.json"
DATA_PATH = Path(os.getenv("TRAFFIC_SIGN_MAPPING_PATH", DEFAULT_DATA_PATH))

_mapping_by_id: dict[int, dict] = {}


def _load_mapping() -> dict[int, dict]:
    with DATA_PATH.open(encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["open_cv_id"]: entry for entry in entries}


_mapping_by_id = _load_mapping()


def lookup(open_cv_id: int) -> dict | None:
    """Returns the mapping entry (stvo_id, value, time_stamp) for a given open_cv_id.

    Returns None if the open_cv_id does not exist in the mapping.
    """
    return _mapping_by_id.get(open_cv_id)


def main():
    print(lookup(2))


if __name__ == "__main__":
    main()
