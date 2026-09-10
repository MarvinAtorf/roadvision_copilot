import re
from pathlib import Path

import pypdf

DEFAULT_PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "StVO.pdf"

PARAGRAPH_PATTERN = re.compile(r"^§\s*\d+[a-z]?\s.*$", re.MULTILINE)


def extract_stvo_text(pdf_path: Path, last_page: int = 38) -> str:
    reader = pypdf.PdfReader(pdf_path)
    pages_text = [page.extract_text() for page in reader.pages[:last_page]]
    return "\n".join(pages_text)


def _is_real_header(header: str) -> bool:
    body = header.split(maxsplit=2)[-1]  # text after "§ N"
    return "Absatz" not in body


def chunk_by_paragraph(text: str) -> list[dict]:
    matches = [m for m in PARAGRAPH_PATTERN.finditer(text) if _is_real_header(m.group().strip())]
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = match.group().strip()
        chunks.append(
            {
                "header": header,
                "text": text[start:end].strip(),
            }
        )
    return chunks
