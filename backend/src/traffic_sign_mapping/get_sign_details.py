"""Single entry point for turning a raw sign-model class_id (0-32) into
full StVO information (official name, explanation, legal sign number).

Usage from the chat/report backend:

    from traffic_sign_mapping.get_sign_details import get_stvo_info

    info = get_stvo_info(7)
    # {
    #   "class_id": 7,
    #   "model_class_name": "RoundaboutMandatory",
    #   "sign_number": "215",
    #   "variant": None,
    #   "official_name": "Kreisverkehr",
    #   "explanation": "Ordnet an, der vorgeschriebenen Fahrtrichtung ...",
    #   "source": "StVO Anlage 2, lfd. Nr. 8",
    # }

This intentionally does a plain dict lookup rather than going through
the ChromaDB/LlamaIndex semantic search in rag/sign_index.py - for an
exact numeric class_id -> sign_number -> explanation chain there is
nothing to search for, a direct lookup is faster and can't return the
wrong sign. The semantic search in rag/sign_index.py remains the right
tool for free-text questions like "what does a red triangle mean".
"""

import json
from pathlib import Path

from traffic_sign_mapping.sign_class_mapping import lookup_by_class_id

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "stvo_sign_catalog.json"

_catalog_by_sign_number: dict[str, dict] | None = None


def _load_catalog() -> dict[str, dict]:
    global _catalog_by_sign_number

    if _catalog_by_sign_number is None:
        if not CATALOG_PATH.exists():
            raise FileNotFoundError(f"StVO sign catalog not found:\n{CATALOG_PATH}")

        with CATALOG_PATH.open(encoding="utf-8") as f:
            entries = json.load(f)

        _catalog_by_sign_number = {entry["sign_number"]: entry for entry in entries}

    return _catalog_by_sign_number


def get_stvo_info(class_id: int) -> dict | None:
    """Look up full StVO info for a raw sign-model class_id.

    Returns None if the class_id is unmapped, or if the mapped sign
    number isn't (yet) present in the catalog - callers should treat
    both the same way (no info available), not raise.
    """
    mapping = lookup_by_class_id(class_id)
    if mapping is None:
        return None

    catalog = _load_catalog()
    catalog_entry = catalog.get(mapping.sign_number)
    if catalog_entry is None:
        return None

    return {
        "class_id": mapping.class_id,
        "model_class_name": mapping.model_class_name,
        "sign_number": mapping.sign_number,
        "variant": mapping.variant,
        "official_name": catalog_entry["official_name"],
        "explanation": catalog_entry["explanation"],
        "source": catalog_entry["source"],
    }