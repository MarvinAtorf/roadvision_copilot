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
    """
    Provides a utility to look up a mapping by an OpenCV identifier.

    The function searches for the given `open_cv_id` within the internal mapping
    and returns the corresponding dictionary if it exists. If no match is found,
    it returns `None`.

    :param open_cv_id: Identifier associated with the desired mapping.
    :type open_cv_id: int
    :return: The dictionary associated with the given identifier, or `None` if
        the identifier was not found.
    :rtype: dict | None
    """

    return _mapping_by_id.get(open_cv_id)


def main():
    print(lookup(2))


if __name__ == "__main__":
    main()
