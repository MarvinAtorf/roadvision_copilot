"""Maps traffic_sign_model.pt's 43 class IDs (standard GTSRB order, from
model.names) to official StVO sign numbers ("Zeichen ...").

Every entry was verified directly against the project's StVO.pdf
(Anlage 1-3), not recalled from memory. This model uses the full
standard 43-class GTSRB taxonomy (speed limits 20-120, etc.) - a
different model from the earlier 33-class traffic_sign_model.pt, so
this mapping is NOT interchangeable with the old one; if the model
weights are swapped, this file must be swapped too.

Some StVO base signs (209, 214, 222, 103) have several pictogram
variants (left/right/straight) sharing the same base Zeichen number -
`variant` records which one this class_id corresponds to.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SignMapping:
    class_id: int
    model_class_name: str
    sign_number: str  # matches "sign_number" in stvo_sign_catalog.json
    variant: str | None
    official_name_de: str
    confidence_note: str | None = None  # set when the mapping isn't fully certain


SIGN_MAPPINGS: list[SignMapping] = [
    SignMapping(0, "speed limit 20", "274-20", None, "Zulässige Höchstgeschwindigkeit 20 km/h"),
    SignMapping(1, "speed limit 30", "274-30", None, "Zulässige Höchstgeschwindigkeit 30 km/h"),
    SignMapping(2, "speed limit 50", "274-50", None, "Zulässige Höchstgeschwindigkeit 50 km/h"),
    SignMapping(3, "speed limit 60", "274-60", None, "Zulässige Höchstgeschwindigkeit 60 km/h"),
    SignMapping(4, "speed limit 70", "274-70", None, "Zulässige Höchstgeschwindigkeit 70 km/h"),
    SignMapping(5, "speed limit 80", "274-80", None, "Zulässige Höchstgeschwindigkeit 80 km/h"),
    SignMapping(
        6, "restriction ends 80", "278-80", None, "Ende der zulässigen Höchstgeschwindigkeit 80 km/h"
    ),
    SignMapping(7, "speed limit 100", "274-100", None, "Zulässige Höchstgeschwindigkeit 100 km/h"),
    SignMapping(8, "speed limit 120", "274-120", None, "Zulässige Höchstgeschwindigkeit 120 km/h"),
    SignMapping(9, "no overtaking", "276", None, "Überholverbot für Kraftfahrzeuge aller Art"),
    SignMapping(
        10, "no overtaking - trucks", "277", None, "Überholverbot für Kraftfahrzeuge über 3,5 t"
    ),
    SignMapping(11, "priority at next intersection", "301", None, "Vorfahrt"),
    SignMapping(12, "priority road", "306", None, "Vorfahrtstraße"),
    SignMapping(13, "give way", "205", None, "Vorfahrt gewähren"),
    SignMapping(14, "stop", "206", None, "Halt. Vorfahrt gewähren."),
    SignMapping(15, "no traffic both ways", "250", None, "Verbot für Fahrzeuge aller Art"),
    SignMapping(16, "no trucks", "253", None, "Verbot für Kraftfahrzeuge über 3,5 t"),
    SignMapping(17, "no entry", "267", None, "Verbot der Einfahrt"),
    SignMapping(18, "danger", "101", None, "Gefahrstelle"),
    SignMapping(19, "bend left", "103", "links", "Kurve"),
    SignMapping(20, "bend right", "103", "rechts", "Kurve"),
    SignMapping(21, "bend", "105", None, "Doppelkurve"),
    SignMapping(22, "uneven road", "112", None, "Unebene Fahrbahn"),
    SignMapping(23, "slippery road", "114", None, "Schleuder- oder Rutschgefahr"),
    SignMapping(24, "road narrows", "121", None, "Einseitig verengte Fahrbahn"),
    SignMapping(25, "construction", "123", None, "Arbeitsstelle"),
    SignMapping(26, "traffic signal", "131", None, "Lichtzeichenanlage"),
    SignMapping(27, "pedestrian crossing", "133", None, "Fußgänger"),
    SignMapping(28, "school crossing", "136", None, "Kinder"),
    SignMapping(29, "cycles crossing", "138", None, "Radverkehr"),
    SignMapping(
        30,
        "snow",
        "101",
        "Zusatzzeichen: Schnee- oder Eisglätte",
        "Gefahrstelle (Schnee-/Eisglätte)",
        confidence_note=(
            "In der aktuellen StVO keine eigene Zeichen-Nummer - "
            "'Schnee- oder Eisglätte' erscheint nur als Zusatzzeichen-Text "
            "zu Zeichen 101, nicht als eigenständiges Anlage-Zeichen. "
            "Vor Produktivnutzung nochmal prüfen."
        ),
    ),
    SignMapping(31, "animals", "142", None, "Wildwechsel"),
    SignMapping(
        32,
        "restriction ends",
        "282",
        None,
        "Ende sämtlicher streckenbezogener Geschwindigkeitsbeschränkungen und Überholverbote",
    ),
    SignMapping(33, "go right", "209", "rechts", "Vorgeschriebene Fahrtrichtung"),
    SignMapping(34, "go left", "209", "links", "Vorgeschriebene Fahrtrichtung"),
    SignMapping(35, "go straight", "209", "geradeaus", "Vorgeschriebene Fahrtrichtung"),
    SignMapping(36, "go right or straight", "214", "rechts", "Geradeaus oder rechts"),
    SignMapping(37, "go left or straight", "214", "links", "Geradeaus oder links"),
    SignMapping(38, "keep right", "222", "rechts", "Vorbeifahrt rechts"),
    SignMapping(39, "keep left", "222", "links", "Vorbeifahrt links"),
    SignMapping(40, "roundabout", "215", None, "Kreisverkehr"),
    SignMapping(
        41, "restriction ends - overtaking", "280", None, "Ende des Überholverbots für Kraftfahrzeuge aller Art"
    ),
    SignMapping(
        42,
        "restriction ends - overtaking trucks",
        "281",
        None,
        "Ende des Überholverbots für Kraftfahrzeuge über 3,5 t",
    ),
]

_BY_CLASS_ID: dict[int, SignMapping] = {m.class_id: m for m in SIGN_MAPPINGS}


def lookup_by_class_id(class_id: int) -> SignMapping | None:
    """Look up the StVO sign number for a raw model class_id (0-42)."""
    return _BY_CLASS_ID.get(class_id)


def sign_number_for(class_id: int) -> str | None:
    """Shortcut: just the sign_number string (e.g. "205"), or None if unmapped."""
    mapping = _BY_CLASS_ID.get(class_id)
    return mapping.sign_number if mapping else None