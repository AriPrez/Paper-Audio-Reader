"""PDF layout extraction and biomedical-safe text cleaning.

The parser keeps line boundaries until the cleaning stage.  This is important:
only a hyphen immediately followed by a line break is considered a word split;
scientific compounds such as ``tumor-specific`` and ``single-cell`` are kept.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

import streamlit as st

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - exercised only in incomplete installs
    fitz = None


KNOWN_SECTION_HEADERS = {
    "abstract",
    "summary",
    "graphical abstract",
    "highlights",
    "in brief",
    "introduction",
    "background",
    "results",
    "discussion",
    "methods",
    "patients and methods",
    "materials and methods",
    "methodology",
    "conclusion",
    "conclusions",
    "limitations",
    "resource availability",
    "lead contact",
    "materials availability",
    "data and code availability",
    "acknowledgments",
    "acknowledgements",
    "author contributions",
    "declaration of interests",
    "supplemental information",
    "star methods",
    "experimental model and study participant details",
    "method details",
    "quantification and statistical analysis",
    "additional resources",
    "key resources table",
    "references",
    "bibliography",
}

BACK_MATTER_HEADERS = {
    "references",
    "bibliography",
    "key resources table",
}


def dehyphenate_text(text: str) -> str:
    """Rejoin words split *across lines* while preserving real compounds.

    Examples:
        ``immuno-\ntherapy`` becomes ``immunotherapy``.
        ``tumor-specific`` and ``single-cell`` remain unchanged.
    """
    if not text:
        return ""

    pattern = r"\b([A-Za-z]{2,})[\-\u00ad\u00ac]\s*\n\s*([A-Za-z]{2,})\b"

    def _rejoin(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if left.isupper() and right.isupper():
            return f"{left}-{right}"
        return f"{left}{right}"

    return re.sub(pattern, _rejoin, text)


def is_running_header_or_footer(block_text: str, bbox: tuple, page_height: float) -> bool:
    """Identify short running headers, footers and page numbers."""
    text = block_text.strip()
    y0, y1 = bbox[1], bbox[3]

    if y0 < 36 or y1 > page_height - 36:
        return True
    if re.fullmatch(r"(?:Page\s+)?\d{1,4}", text, flags=re.IGNORECASE):
        return True
    normalised = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    if normalised in {"article", "cell article", "ll article", "report", "open access"}:
        return True
    return False


def is_front_matter_or_author_block(block_text: str, page_num: int) -> bool:
    """Filter compact affiliation/author metadata on the first two pages.

    Keyword matching is deliberately limited to short early-page blocks so a
    methods paragraph mentioning a hospital or institute is never discarded.
    """
    text = block_text.strip()
    if not text:
        return True
    lowered = text.lower()

    if lowered in {"article", "cell article", "report", "open access"}:
        return True
    if page_num > 2 or len(text.split()) > 90:
        return False
    if "@" in text or "doi.org" in lowered or lowered.startswith("doi:"):
        return True
    metadata_terms = (
        "department of",
        "school of medicine",
        "correspondence:",
        "lead contact:",
        "these authors contributed",
        "affiliation",
    )
    return any(term in lowered for term in metadata_terms)


def is_standalone_caption_or_table_data(
    block_text: str,
    block_bbox: tuple,
    table_bboxes: Iterable[tuple],
) -> bool:
    """Identify figure/table captions and blocks inside detected tables."""
    text = block_text.strip()
    if not text:
        return True

    bx0, by0, bx1, by1 = block_bbox
    for tx0, ty0, tx1, ty1 in table_bboxes:
        if not (bx1 < tx0 or bx0 > tx1 or by1 < ty0 or by0 > ty1):
            return True

    if re.match(r"^(?:Figure|Fig\.|Table|Tab\.)\s+[A-Z]?\d+\b", text, re.IGNORECASE):
        return True

    digits = sum(char.isdigit() for char in text)
    letters = sum(char.isalpha() for char in text)
    if digits >= 3 and len(text.split()) <= 8 and not text.endswith((".", ":")):
        return True
    return digits > 8 and digits > letters * 1.5 and len(text.split()) < 24


def is_likely_figure_label(block_text: str, dense_page: bool = False) -> bool:
    """Detect short labels embedded in complex multi-panel figures.

    Vector figure labels often look bold and large to PDF extractors, so font
    size alone cannot distinguish them from headings. On pages containing many
    separate text blocks, short fragments without sentence punctuation are much
    more likely to belong to a panel than to the narrative.
    """
    text = re.sub(r"\s+", " ", block_text).strip()
    normalised = _normalise_header(text)
    if normalised in KNOWN_SECTION_HEADERS:
        return False
    if re.search(r"(?:legend on next page|recist response|input t cells)", text, re.IGNORECASE):
        return True
    if any(symbol in text for symbol in ("↘", "→", "○", "●")):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z+\-]*", text)
    has_sentence_end = text.endswith((".", "?", "!", ":"))
    if dense_page and len(words) <= 10 and not has_sentence_end and len(text) <= 90:
        return True
    if len(words) <= 4 and not has_sentence_end:
        return bool(re.search(r"\d|\[|\]|\+|\-|\b(?:high|low|mid|strong|weak)\b", text, re.IGNORECASE))
    return False


def sort_blocks_2column_order(blocks: list, page_width: float) -> list:
    """Sort mixed full-width and two-column blocks in reading order.

    Full-width blocks split the page into vertical bands. Within each band the
    left column is read top-to-bottom, followed by the right column. This avoids
    moving a full-width block at the bottom of the page before all body text.
    """
    if not blocks:
        return []

    full_width = []
    narrow = []
    for block in blocks:
        x0, y0, x1, y1 = block["bbox"]
        if x1 - x0 >= page_width * 0.68:
            full_width.append(block)
        else:
            narrow.append(block)

    full_width.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    pending = set(range(len(narrow)))
    ordered: list = []
    band_top = float("-inf")

    def append_band(band_bottom: float) -> None:
        candidates = []
        for index in list(pending):
            block = narrow[index]
            _, y0, _, y1 = block["bbox"]
            center_y = (y0 + y1) / 2
            if band_top <= center_y < band_bottom:
                candidates.append((index, block))

        left = []
        right = []
        midpoint = page_width / 2
        for index, block in candidates:
            x0, _, x1, _ = block["bbox"]
            (left if (x0 + x1) / 2 < midpoint else right).append(block)
            pending.remove(index)
        left.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        right.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        ordered.extend(left)
        ordered.extend(right)

    for block in full_width:
        append_band(block["bbox"][1])
        ordered.append(block)
        band_top = max(band_top, block["bbox"][3])

    append_band(float("inf"))
    if pending:  # Overlapping blocks that straddle a full-width boundary.
        leftovers = [narrow[index] for index in pending]
        leftovers.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        ordered.extend(leftovers)
    return ordered


def _looks_biomedical(text: str) -> bool:
    """Return True for common marker/cytokine notation that is not an equation."""
    return bool(
        re.search(
            r"\b(?:CD\d+|IFN|TNF|IL-?\d*|TGF|CXCL\d*|CXCR\d*|PD-?1|CTLA-?4)"
            r"(?:[-–]?[\u03b1-\u03c9\u0391-\u03a9])?\+?\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def is_equation_line(line: str) -> bool:
    """Detect standalone equations without mistaking immune markers for math."""
    text = line.strip()
    if not text or _looks_biomedical(text):
        return False

    if re.search(r"(?:\bEq\.|\bEquation)\s*\d+", text, re.IGNORECASE):
        return True
    if re.search(r"\\(?:sum|int|frac|sqrt|mathbb|mathcal)\b", text):
        return True

    equation_operators = len(re.findall(r"(?:=|≤|≥|≠|≈|∑|∏|∫|√|∝)", text))
    word_count = len(re.findall(r"[A-Za-z]{2,}", text))
    if equation_operators >= 2 and word_count < 10:
        return True
    if equation_operators >= 1 and word_count <= 3 and re.search(r"[A-Za-z0-9]\s*=", text):
        return True
    return False


def remove_parenthetical_citations(text: str) -> str:
    """Remove author-year and numeric citations enclosed in parentheses.

    Scientific content such as ``(IFN-γ+)``, ``(n = 24)`` or confidence
    intervals is preserved. Nested parentheses are intentionally left alone
    because they are more likely to contain scientific notation than citations.
    """
    numeric_citation = re.compile(
        r"\s*\d+[A-Za-z]?(?:\s*[-–—,;]\s*\d+[A-Za-z]?)*\s*"
    )
    referenced_numbers = re.compile(
        r"\s*(?:refs?|references?)\.?\s*\d+"
        r"(?:\s*[-–—,;]\s*\d+)*\s*",
        flags=re.IGNORECASE,
    )
    year = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b", flags=re.IGNORECASE)
    author_signal = re.compile(
        r"\bet\s+al\.?\b|\b[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'’-]{2,}\b",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        payload = match.group(1).strip()
        if numeric_citation.fullmatch(payload) or referenced_numbers.fullmatch(payload):
            return ""
        if year.search(payload) and author_signal.search(payload):
            # Avoid treating experimental metadata containing a year as a cite.
            if re.search(r"\b(?:n|p)\s*[=<>]", payload, flags=re.IGNORECASE):
                return match.group(0)
            return ""
        return match.group(0)

    return re.sub(r"\(([^()]*)\)", replace, text)


def remove_unicode_superscript_citations(text: str) -> str:
    """Remove citation-like Unicode superscripts without harming science.

    Real PDF superscripts are normally detected from glyph geometry before
    this cleaning stage. This fallback covers documents that encode citation
    numbers as Unicode characters such as ``¹`` and ``²–⁴``. A citation must
    follow a word of at least three letters or sentence punctuation. This keeps
    common scientific forms such as ``10⁶``, ``m²``, ``x²`` and ``Ca²⁺``.
    """
    superscript_digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    pattern = re.compile(
        rf"(?P<prefix>\b[A-Za-zÀ-ÖØ-öø-ÿ]{{3,}}|[.,;:!?\)\]\}}])"
        rf"\s*(?:[{superscript_digits}]+"
        rf"(?:\s*[,;–—-]\s*[{superscript_digits}]+)*)"
        rf"(?![{superscript_digits}⁺⁻+−±%])"
    )
    return pattern.sub(r"\g<prefix>", text)


def clean_academic_text(
    text: str,
    filter_brackets: bool = True,
    filter_parentheses: bool = True,
    filter_superscript_citations: bool = True,
    filter_urls: bool = True,
    filter_captions: bool = True,
    filter_equations_flag: bool = True,
) -> str:
    """Prepare extracted academic text for speech."""
    if not text:
        return ""

    cleaned = dehyphenate_text(text)
    kept_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if filter_captions and re.match(
            r"^(?:Figure|Fig\.|Table|Tab\.)\s+[A-Z]?\d+\b", stripped, re.IGNORECASE
        ):
            continue
        if filter_equations_flag and is_equation_line(stripped):
            continue
        kept_lines.append(stripped)
    cleaned = "\n".join(kept_lines)

    if filter_urls:
        cleaned = re.sub(r"https?://[^\s)]+|www\.[^\s)]+", "", cleaned)
        cleaned = re.sub(
            r"\b(?:doi:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    if filter_brackets:
        cleaned = re.sub(
            r"\[\s*\d+[A-Za-z]?(?:\s*[-–—,;]\s*\d+[A-Za-z]?)*\s*\]",
            "",
            cleaned,
        )

    if filter_parentheses:
        cleaned = remove_parenthetical_citations(cleaned)

    if filter_superscript_citations:
        cleaned = remove_unicode_superscript_citations(cleaned)

    if filter_equations_flag:
        cleaned = re.sub(r"\$[^$]+\$", "", cleaned)

    cleaned = re.sub(r"[ \t]+([,.;:?!])", r"\1", cleaned)
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"([.!?])(?=[A-ZÀ-ÖØ-Ý])", r"\1 ", cleaned)
    return cleaned.strip()


def _normalise_header(text: str) -> str:
    text = text.replace("★", " ")
    text = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", text)
    return re.sub(r"\s+", " ", text).strip().rstrip(".:").lower()


def is_true_section_header(
    text: str,
    font_size: float | None = None,
    body_font_size: float | None = None,
    is_bold: bool = False,
) -> bool:
    """Classify a section heading using content and PDF typography."""
    raw = re.sub(r"\s+", " ", text).strip()
    if not raw or len(raw) > 110 or len(raw.split()) > 14:
        return False
    if any(ord(char) < 32 for char in raw):
        return False

    lowered = _normalise_header(raw)
    if lowered in KNOWN_SECTION_HEADERS:
        return True

    alpha_words = re.findall(r"[A-Za-z]{2,}", raw)
    digit_count = sum(char.isdigit() for char in raw)
    alpha_count = sum(char.isalpha() for char in raw)
    statistical = bool(
        re.search(r"\bP\s*[<=>]", raw, re.IGNORECASE)
        or "%" in raw
        or (digit_count >= 3 and digit_count >= alpha_count / 2)
    )
    if statistical or len(alpha_words) < 2 or raw.endswith((";", ",")):
        return False

    numbered = bool(re.match(r"^\s*\d+(?:\.\d+)*[.)]\s+", raw))
    if re.match(r"^\s*\d+\s+", raw) and not numbered:
        return False
    typography_available = font_size is not None and body_font_size is not None
    typographic_heading = bool(
        typography_available
        and (
            font_size >= body_font_size * 1.16
            or (is_bold and font_size >= body_font_size * 0.98)
        )
    )

    if numbered and (typographic_heading or not typography_available):
        return True
    # Generic typographic headings are intentionally not accepted. Multi-panel
    # figure labels frequently look larger/bolder than body text. Keeping an
    # ambiguous subheading inside its parent section is safer than splitting the
    # paper at a chart label.
    return False


def _join_positioned_chars(spans: list[dict]) -> str:
    """Rebuild a line from glyph positions, including omitted PDF spaces."""
    characters = []
    for span in spans:
        size = float(span.get("size", 0.0))
        for char in span.get("chars", []):
            value = char.get("c", "")
            bbox = char.get("bbox")
            if value and bbox:
                characters.append({"text": value, "bbox": bbox, "size": size})
    characters.sort(key=lambda char: char["bbox"][0])

    result = ""
    previous = None
    for char in characters:
        value = char["text"]
        if value.isspace():
            if result and not result[-1].isspace():
                result += " "
            previous = char
            continue

        if previous is not None and result and not result[-1].isspace():
            gap = float(char["bbox"][0]) - float(previous["bbox"][2])
            average_size = (float(previous["size"]) + float(char["size"])) / 2
            # Cell/Elsevier PDFs can encode visible word gaps as only ~14% of
            # the font size while omitting the actual space glyph. Kerning gaps
            # inside words remain near zero, so 10% separates the two reliably.
            minimum_space = max(0.50, average_size * 0.10)
            punctuation = value in ",.;:?!)]}%'’"
            opening = result[-1] in "([{/'’-–—"
            if gap >= minimum_space and not punctuation and not opening:
                result += " "
        result += value
        previous = char
    return result.strip()


def _span_text(span: dict) -> str:
    value = span.get("text")
    if value is not None:
        return value
    return "".join(char.get("c", "") for char in span.get("chars", []))


def _citation_like_superscript_spans(spans: list[dict]) -> set[int]:
    """Return indices of raised numeric spans that look bibliographic.

    PyMuPDF's superscript flag is preferred. A size/baseline fallback supports
    publisher PDFs that only encode the visual position. Context guards avoid
    removing exponents attached to numbers, one/two-letter variables or units,
    and charged scientific notation.
    """
    if not spans:
        return set()

    texts = [_span_text(span) for span in spans]
    size_counts: Counter = Counter()
    for span, value in zip(spans, texts):
        size = round(float(span.get("size", 0.0)), 1)
        if size > 0 and value.strip():
            size_counts[size] += max(1, len(value.strip()))
    if not size_counts:
        return set()
    body_size = float(max(size_counts, key=lambda size: (size_counts[size], size)))

    body_origins = []
    for span, value in zip(spans, texts):
        origin = span.get("origin")
        size = float(span.get("size", 0.0))
        if value.strip() and origin and size >= body_size * 0.90:
            body_origins.extend([float(origin[1])] * max(1, len(value.strip())))
    if body_origins:
        ordered_origins = sorted(body_origins)
        body_baseline = ordered_origins[len(ordered_origins) // 2]
    else:
        body_baseline = None

    allowed_fragment = re.compile(r"^[\s\dA-Za-z,;–—-]+$")
    citation_group = re.compile(
        r"^\s*\d+[A-Za-z]?(?:\s*[,;–—-]\s*\d+[A-Za-z]?)*\s*$"
    )

    def visually_raised(index: int) -> bool:
        span = spans[index]
        if int(span.get("flags", 0)) & 1:
            return True
        size = float(span.get("size", 0.0))
        origin = span.get("origin")
        return bool(
            body_baseline is not None
            and origin
            and 0 < size <= body_size * 0.84
            and float(origin[1]) <= body_baseline - max(0.5, body_size * 0.10)
        )

    candidates = [
        bool(value.strip() and allowed_fragment.fullmatch(value) and visually_raised(index))
        for index, value in enumerate(texts)
    ]
    removable: set[int] = set()
    index = 0
    while index < len(spans):
        if not candidates[index]:
            index += 1
            continue
        end = index + 1
        while end < len(spans) and candidates[end]:
            end += 1
        payload = "".join(texts[index:end])
        if citation_group.fullmatch(payload):
            prefix = "".join(texts[:index]).rstrip()
            suffix = "".join(texts[end:]).lstrip()
            previous_token = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ]+|\d+)$", prefix)
            follows_scientific_token = bool(
                previous_token
                and (
                    previous_token.group(1).isdigit()
                    or len(previous_token.group(1)) <= 2
                )
            )
            has_charge_or_unit_suffix = bool(suffix[:1] in "+⁺−⁻±%")
            has_citation_context = bool(
                prefix
                and (
                    prefix[-1] in ".,;:!?)]}"
                    or (
                        previous_token
                        and previous_token.group(1).isalpha()
                        and len(previous_token.group(1)) >= 3
                    )
                )
            )
            if has_citation_context and not follows_scientific_token and not has_charge_or_unit_suffix:
                removable.update(range(index, end))
        index = end
    return removable


def join_line_spans(spans: list[dict], filter_superscript_citations: bool = False) -> str:
    """Rebuild spaces inside and between PDF spans from glyph geometry."""
    if filter_superscript_citations:
        removable = _citation_like_superscript_spans(spans)
        spans = [span for index, span in enumerate(spans) if index not in removable]
    if any(span.get("chars") for span in spans):
        return _join_positioned_chars(spans)

    ordered = sorted(spans, key=lambda span: span.get("bbox", (0, 0, 0, 0))[0])
    result = ""
    previous = None
    for span in ordered:
        value = span.get("text", "")
        if not value:
            continue
        if previous is not None and result and not result[-1].isspace() and not value[0].isspace():
            previous_bbox = previous.get("bbox", (0, 0, 0, 0))
            current_bbox = span.get("bbox", (0, 0, 0, 0))
            gap = float(current_bbox[0]) - float(previous_bbox[2])
            average_size = (
                float(previous.get("size", 0.0)) + float(span.get("size", 0.0))
            ) / 2
            minimum_space = max(0.45, average_size * 0.10)
            punctuation = value[0] in ",.;:?!)]}"
            opening = result[-1] in "([{/-–—"
            if gap >= minimum_space and not punctuation and not opening:
                result += " "
        result += value
        previous = span
    return result.strip()


def _block_metadata(block: dict) -> dict | None:
    lines_text = []
    line_bboxes = []
    line_items = []
    span_sizes = []
    bold_chars = 0
    total_chars = 0

    for line in block.get("lines", []):
        spans = line.get("spans", [])
        for span in spans:
            value = span.get("text")
            if value is None:
                value = "".join(char.get("c", "") for char in span.get("chars", []))
            char_count = len(value.strip())
            if char_count:
                size = float(span.get("size", 0.0))
                span_sizes.extend([round(size, 1)] * char_count)
                total_chars += char_count
                font_name = span.get("font", "").lower()
                if span.get("flags", 0) & 16 or "bold" in font_name:
                    bold_chars += char_count
        line_text = join_line_spans(spans)
        speech_line_text = join_line_spans(spans, filter_superscript_citations=True)
        if line_text:
            lines_text.append(line_text)
            line_bbox = tuple(line.get("bbox"))
            line_bboxes.append(line_bbox)
            line_items.append(
                {
                    "text": line_text,
                    "text_without_superscript_citations": speech_line_text,
                    "bbox": line_bbox,
                }
            )

    text = "\n".join(lines_text).strip()
    if not text:
        return None
    font_size = max(span_sizes) if span_sizes else 0.0
    return {
        "bbox": tuple(block.get("bbox")),
        "line_bboxes": line_bboxes,
        "lines": line_items,
        "text": text,
        "font_size": font_size,
        "is_bold": total_chars > 0 and bold_chars / total_chars >= 0.55,
        "font_samples": span_sizes,
    }


def _body_font_size(blocks: list[dict]) -> float:
    samples = []
    for block in blocks:
        samples.extend(block.get("font_samples", []))
    if not samples:
        return 10.0
    counts = Counter(samples)
    return float(max(counts, key=lambda size: (counts[size], -abs(size - 10.0))))


@st.cache_data(show_spinner=False)
def extract_page_blocks_for_selection(pdf_bytes: bytes, page_num: int) -> dict:
    """Return ordered text blocks and page dimensions for rectangle selection."""
    if fitz is None:
        raise ImportError("PyMuPDF is required for PDF extraction.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Page {page_num} is outside 1-{len(doc)}.")
        page = doc[page_num - 1]
        blocks = []
        # rawdict exposes individual glyph boxes. Many publisher PDFs omit
        # actual space characters, so these positions are required to rebuild
        # visually obvious word boundaries.
        for raw in page.get_text("rawdict").get("blocks", []):
            if raw.get("type") != 0:
                continue
            metadata = _block_metadata(raw)
            if metadata and len(metadata["text"].strip()) >= 2:
                blocks.append(metadata)
        return {
            "width": float(page.rect.width),
            "height": float(page.rect.height),
            "blocks": sort_blocks_2column_order(blocks, float(page.rect.width)),
        }
    finally:
        doc.close()


def select_blocks_in_region(
    blocks: list[dict],
    page_width: float,
    page_height: float,
    region: dict | None,
) -> list[dict]:
    """Return blocks intersecting a normalized rectangle by at least 10%."""
    if not region:
        return []
    try:
        x0 = max(0.0, min(1.0, float(region["x0"]))) * page_width
        y0 = max(0.0, min(1.0, float(region["y0"]))) * page_height
        x1 = max(0.0, min(1.0, float(region["x1"]))) * page_width
        y1 = max(0.0, min(1.0, float(region["y1"]))) * page_height
    except (KeyError, TypeError, ValueError):
        return []
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return []

    selected = []
    for block in blocks:
        bx0, by0, bx1, by1 = block["bbox"]
        intersection = max(0.0, min(x1, bx1) - max(x0, bx0)) * max(
            0.0, min(y1, by1) - max(y0, by0)
        )
        block_area = max(1.0, (bx1 - bx0) * (by1 - by0))
        center_inside = x0 <= (bx0 + bx1) / 2 <= x1 and y0 <= (by0 + by1) / 2 <= y1
        if center_inside or intersection / block_area >= 0.10:
            selected.append(block)
    return selected


def order_selected_text(
    blocks: list[dict],
    page_width: float,
    page_height: float,
    region: dict | None,
    layout_mode: str = "automatic",
    filter_superscript_citations: bool = False,
) -> str:
    """Order selected text as automatic, one-column or forced two-column.

    In two-column mode the midpoint of the user's rectangle is the gutter:
    every line in the left half is read top-to-bottom before the right half.
    Lines are filtered against the rectangle so a partially intersecting PDF
    block cannot bring unrelated text from outside the selection.
    """
    if not blocks or not region:
        return ""
    if layout_mode not in {"automatic", "single_column", "two_columns"}:
        raise ValueError(f"Unknown layout mode: {layout_mode}")

    try:
        x0 = max(0.0, min(1.0, float(region["x0"]))) * page_width
        y0 = max(0.0, min(1.0, float(region["y0"]))) * page_height
        x1 = max(0.0, min(1.0, float(region["x1"]))) * page_width
        y1 = max(0.0, min(1.0, float(region["y1"]))) * page_height
    except (KeyError, TypeError, ValueError):
        return ""
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    grouped_lines = []
    all_lines = []
    for block_index, block in enumerate(blocks):
        block_lines = []
        for line_index, line in enumerate(block.get("lines", [])):
            lx0, ly0, lx1, ly1 = line["bbox"]
            intersection = max(0.0, min(x1, lx1) - max(x0, lx0)) * max(
                0.0, min(y1, ly1) - max(y0, ly0)
            )
            line_area = max(1.0, (lx1 - lx0) * (ly1 - ly0))
            center_inside = x0 <= (lx0 + lx1) / 2 <= x1 and y0 <= (ly0 + ly1) / 2 <= y1
            if center_inside or intersection / line_area >= 0.10:
                record = {
                    "text": (
                        line.get("text_without_superscript_citations", line["text"])
                        if filter_superscript_citations
                        else line["text"]
                    ),
                    "bbox": line["bbox"],
                    "block_index": block_index,
                    "line_index": line_index,
                }
                block_lines.append(record)
                all_lines.append(record)
        if block_lines:
            grouped_lines.append(block_lines)

    if not all_lines:
        return ""
    if layout_mode == "automatic":
        return "\n\n".join(
            "\n".join(line["text"] for line in group) for group in grouped_lines
        )
    if layout_mode == "single_column":
        ordered = sorted(all_lines, key=lambda line: (line["bbox"][1], line["bbox"][0]))
        return "\n".join(line["text"] for line in ordered)

    split_x = (x0 + x1) / 2
    left = []
    right = []
    for line in all_lines:
        lx0, _, lx1, _ = line["bbox"]
        (left if (lx0 + lx1) / 2 < split_x else right).append(line)
    left.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    right.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    columns = ["\n".join(line["text"] for line in column) for column in (left, right) if column]
    return "\n\n".join(columns)


@st.cache_data(show_spinner=False)
def extract_sections_with_bboxes(
    pdf_bytes: bytes,
    filter_brackets: bool = True,
    filter_parentheses: bool = True,
    filter_urls: bool = True,
    filter_captions: bool = True,
    filter_equations_flag: bool = True,
    include_back_matter: bool = False,
):
    """Extract body sections with typography-aware headings and bboxes."""
    if fitz is None:
        raise ImportError("PyMuPDF is required for PDF extraction.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sections = []
    current_title = "Opening / Summary"
    current_blocks: list[str] = []
    current_bboxes: dict[int, list] = {}
    page_start = 1
    section_id = 1

    def flush_section() -> None:
        nonlocal section_id
        if not current_blocks:
            return
        raw_text = "\n\n".join(current_blocks)
        clean_text = clean_academic_text(
            raw_text,
            filter_brackets=filter_brackets,
            filter_parentheses=filter_parentheses,
            filter_urls=filter_urls,
            filter_captions=filter_captions,
            filter_equations_flag=filter_equations_flag,
        )
        if len(clean_text.split()) < 2:
            return
        sections.append(
            {
                "id": section_id,
                "title": current_title,
                "page_start": page_start,
                "bboxes_by_page": {page: items.copy() for page, items in current_bboxes.items()},
                "raw_text": raw_text,
                "clean_text": clean_text,
            }
        )
        section_id += 1

    skipping_back_matter = False
    try:
        for page_num in range(1, len(doc) + 1):
            page = doc[page_num - 1]
            rect = page.rect
            crop_box = fitz.Rect(0, 36, rect.width, rect.height - 36)

            table_bboxes = []
            if filter_captions:
                try:
                    tables = page.find_tables(clip=crop_box)
                    table_bboxes = [table.bbox for table in getattr(tables, "tables", tables or [])]
                except Exception:
                    table_bboxes = []

            blocks = []
            for raw in page.get_text("dict", clip=crop_box).get("blocks", []):
                if raw.get("type") != 0:
                    continue
                metadata = _block_metadata(raw)
                if metadata:
                    blocks.append(metadata)
            body_font = _body_font_size(blocks)
            dense_page = len(blocks) >= 18

            for block in sort_blocks_2column_order(blocks, float(rect.width)):
                text = block["text"].strip()
                bbox = block["bbox"]
                if len(text) < 2 or is_running_header_or_footer(text, bbox, float(rect.height)):
                    continue
                if is_front_matter_or_author_block(text, page_num):
                    continue
                if filter_captions and is_standalone_caption_or_table_data(text, bbox, table_bboxes):
                    continue
                if filter_captions and is_likely_figure_label(text, dense_page=dense_page):
                    continue

                heading = is_true_section_header(
                    text,
                    font_size=block["font_size"],
                    body_font_size=body_font,
                    is_bold=block["is_bold"],
                )
                normalised = _normalise_header(text)
                if heading and not include_back_matter and normalised in BACK_MATTER_HEADERS:
                    flush_section()
                    current_blocks.clear()
                    current_bboxes.clear()
                    skipping_back_matter = True
                    continue

                if skipping_back_matter:
                    if not heading:
                        continue
                    skipping_back_matter = False

                if heading:
                    flush_section()
                    current_title = re.sub(r"\s+", " ", text).strip().replace("★", " ").title()
                    current_blocks.clear()
                    current_bboxes.clear()
                    page_start = page_num

                current_blocks.append(text)
                current_bboxes.setdefault(page_num, []).append(
                    {"block_bbox": bbox, "line_bboxes": block["line_bboxes"]}
                )

        flush_section()
    finally:
        doc.close()
    return sections


@st.cache_data(show_spinner=False)
def render_page_with_red_underlines(
    pdf_bytes: bytes,
    page_num: int,
    bboxes: list,
    dpi: int = 150,
) -> bytes:
    """Render a page and underline selected lines in red."""
    if fitz is None:
        raise ImportError("PyMuPDF is required for page rendering.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Page {page_num} is outside 1-{len(doc)}.")
        page = doc[page_num - 1]
        shape = page.new_shape()
        for item in bboxes:
            line_bboxes = item.get("line_bboxes", []) if isinstance(item, dict) else [item]
            for line_bbox in line_bboxes:
                x0, y0, x1, y1 = line_bbox
                if y0 >= 36:
                    shape.draw_line(fitz.Point(x0, y1 - 1), fitz.Point(x1, y1 - 1))
                    shape.finish(color=(0.95, 0.1, 0.1), width=1.8)
        shape.commit()
        return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")
    finally:
        doc.close()
