import re
from pathlib import Path

import pypdf

DEFAULT_PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "bussgeldkatalog.pdf"

# Repeated ad line, split across two physical lines in the PDF's own text
# order (not a single line ending in "+++" — the two halves never share a
# line), so two separate patterns instead of one combined pattern.
AD_LINE = re.compile(r"^\+\+\+.*$|^Möglichkeiten auf www\.sos-verkehrsrecht\.de.*$", re.MULTILINE)

# Recurring page furniture: page number line ("Seite |  N", note the double
# space before the digit — extra whitespace in the source, not a typo here),
# breadcrumb, running title, TOC-jump link, footer brand line.
NAV_LINE = re.compile(
    r"^(Seite\s*\|\s*\d+.*|Bu[ßss]geldkatalog für.*|DER AKTUELLE BU.?[ßss]GELDKATALOG.*"
    r"|.?\s*zum Inhaltsverzeichnis.*|Bussgeldkatalog\.org)$",
    re.MULTILINE,
)

# A category footer tag: a line made up only of uppercase letters/umlauts/ß/
# "&"/hyphen/literal spaces. Deliberately NOT \s here — \s also matches
# newline, which let the pattern jump across blank lines and swallow the
# next real content by mistake.
CATEGORY_LINE = re.compile(r"^[A-ZÄÖÜß&\- ]{3,50}$", re.MULTILINE)


def extract_bussgeld_text(pdf_path: Path, first_page: int = 3, last_page: int = 39) -> str:
    """
    Extract and clean the Pkw section of the Bußgeldkatalog PDF.

    Defaults bound the extraction to exactly the Pkw chapter (pages index
    3-38: "Abstand" through "Vorfahrt") — cover, foreword and table of
    contents (index 0-2) and the Lkw/Rad/E-Scooter chapters (index 39+) are
    excluded by these bounds, not filtered out afterwards.

    :param pdf_path: Path to the Bußgeldkatalog PDF.
    :param first_page: First page index (inclusive) to extract.
    :param last_page: Last page index (exclusive) to extract.
    :return: Cleaned, concatenated text of the selected pages.
    :rtype: str
    """
    reader = pypdf.PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages[first_page:last_page]:
        raw = page.extract_text()
        cleaned = AD_LINE.sub("", raw)
        cleaned = NAV_LINE.sub("", cleaned)
        # Drop now-empty lines and the lone "!" callout-icon glyph left
        # over from "Achtung!" boxes once their line's other text is gone.
        lines = [line for line in cleaned.split("\n") if line.strip() not in ("", "!")]
        pages_text.append("\n".join(lines))
    return "\n".join(pages_text)


def chunk_by_bussgeld_category(text: str) -> list[dict]:
    """
    Split the cleaned Pkw-section text into one chunk per category.

    The category name is a footer tag that trails its own content and
    repeats on every page belonging to that category (unlike ingest_stvo's
    § headers, which appear once, in the text, at the start of their
    section) — so consecutive matches with the SAME name are merged into
    one run before a chunk boundary is drawn, and each chunk ends at
    (not starts at) its category tag.

    :param text: Cleaned text, as returned by extract_bussgeld_text().
    :return: A list of {"category": str, "text": str} dicts, one per
        Bußgeldkatalog category.
    :rtype: list[dict]
    """
    matches = list(CATEGORY_LINE.finditer(text))

    runs = []
    for match in matches:
        if runs and runs[-1]["name"] == match.group():
            runs[-1]["matches"].append(match)
        else:
            runs.append({"name": match.group(), "matches": [match]})

    chunks = []
    chunk_start = 0
    for run in runs:
        chunk_end = run["matches"][-1].end()
        chunks.append(
            {
                "category": run["name"],
                "text": text[chunk_start:chunk_end].strip(),
            }
        )
        chunk_start = chunk_end

    return chunks
