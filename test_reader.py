"""Offline regression tests for the academic paper reader."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from parser import (  # noqa: E402
    clean_academic_text,
    dehyphenate_text,
    extract_page_blocks_for_selection,
    extract_sections_with_bboxes,
    is_true_section_header,
    join_line_spans,
    normalized_block_boxes,
    order_selected_text,
    remove_parenthetical_citations,
    remove_unicode_superscript_citations,
    render_page_with_red_underlines,
    select_blocks_in_region,
)
from tts_engine import (  # noqa: E402
    DEFAULT_CHUNK_CHARS,
    chunk_text,
    estimate_mp3_duration,
    generate_speech_async,
)


def create_test_academic_pdf() -> bytes:
    """Create a small, genuinely multiline two-page paper fixture."""
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_textbox(fitz.Rect(54, 55, 558, 85), "Abstract", fontsize=14, fontname="hebo")
    page.insert_textbox(
        fitz.Rect(54, 90, 558, 155),
        "Tumor-specific tissue-resident cells were quantified (Smith et al., 2020) [1-3]. "
        "IFN-γ+ TNF-α+ cells remained detectable.",
        fontsize=10,
    )
    page.insert_textbox(fitz.Rect(54, 175, 558, 205), "1. Introduction", fontsize=14, fontname="hebo")
    page.insert_textbox(
        fitz.Rect(54, 210, 290, 360),
        "Single-cell profiling identifies immune populations. Immuno-\ntherapy responses vary between patients.",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(322, 210, 558, 360),
        "The right column follows the complete left column and preserves biological terminology.",
        fontsize=10,
    )

    page = document.new_page(width=612, height=792)
    page.insert_textbox(fitz.Rect(54, 55, 558, 85), "2. Results", fontsize=14, fontname="hebo")
    page.insert_textbox(
        fitz.Rect(54, 90, 290, 230),
        "The treatment increased the abundance of CXCR6+ CD8+ cells. The effect was reproducible.",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(322, 90, 558, 130),
        "96 Conditions 335,917 Cells",
        fontsize=14,
        fontname="hebo",
    )
    page.insert_textbox(fitz.Rect(322, 135, 558, 170), "P = 0.0059", fontsize=14, fontname="hebo")
    page.insert_textbox(
        fitz.Rect(322, 180, 558, 240),
        "Figure 1. Quantification of activated populations across conditions.",
        fontsize=9,
    )
    page.insert_textbox(fitz.Rect(54, 270, 558, 300), "Discussion", fontsize=14, fontname="hebo")
    page.insert_textbox(
        fitz.Rect(54, 305, 558, 380),
        "These observations support a durable immune response without changing scientific compounds.",
        fontsize=10,
    )
    page.insert_textbox(fitz.Rect(54, 420, 558, 450), "References", fontsize=14, fontname="hebo")
    page.insert_textbox(
        fitz.Rect(54, 455, 558, 500),
        "Smith A. Example reference. 2020.",
        fontsize=9,
    )

    result = document.tobytes()
    document.close()
    return result


def test_dehyphenation_only_joins_line_breaks() -> None:
    text = "tumor-specific single-cell immuno-\ntherapy MD-VRP"
    assert dehyphenate_text(text) == "tumor-specific single-cell immunotherapy MD-VRP"


def test_biomedical_cleaning_preserves_markers_and_filters_citations() -> None:
    raw = (
        "Tumor-specific tissue-resident single-cell response.\n"
        "IFN-γ+ TNF-α+ cells (Smith et al., 2020; Doe et al., 2021) [1, 3-5]."
    )
    cleaned = clean_academic_text(raw)
    assert "Tumor-specific" in cleaned
    assert "tissue-resident" in cleaned
    assert "single-cell" in cleaned
    assert "IFN-γ+ TNF-α+" in cleaned
    assert "Smith" not in cleaned
    assert "[1" not in cleaned


@pytest.mark.parametrize(
    "citation",
    [
        "(Smith et al., 2020; Doe and Roe, 2021)",
        "(van der Woude et al. 2019)",
        "(1)",
        "(1–3, 7)",
        "(refs. 2-5)",
    ],
)
def test_parenthetical_citation_variants_are_removed(citation: str) -> None:
    assert remove_parenthetical_citations(f"Before {citation} after") == "Before  after"


@pytest.mark.parametrize("content", ["(IFN-γ+)", "(n = 24)", "(95% CI, 1.2–2.4)"])
def test_scientific_parentheses_are_preserved(content: str) -> None:
    assert remove_parenthetical_citations(content) == content


def test_each_filter_can_be_disabled() -> None:
    raw = "Figure 2. Result overview\nText (Smith et al., 2020) [2] https://example.org"
    untouched = clean_academic_text(
        raw,
        filter_brackets=False,
        filter_parentheses=False,
        filter_urls=False,
        filter_captions=False,
    )
    assert "Figure 2" in untouched
    assert "Smith" in untouched
    assert "[2]" in untouched
    assert "https://example.org" in untouched
    filtered = clean_academic_text(raw)
    assert "Figure 2" not in filtered
    assert "Smith" not in filtered
    assert "[2]" not in filtered
    assert "example.org" not in filtered


@pytest.mark.parametrize(
    "label",
    ["96 Conditions 335,917 Cells", "64.0% 2.2%", "13.5% 83.9%", "P = 0.0059"],
)
def test_statistics_are_not_section_headers(label: str) -> None:
    assert not is_true_section_header(label, font_size=14, body_font_size=10, is_bold=True)


def test_section_extraction_uses_typography_and_stops_before_references() -> None:
    sections = extract_sections_with_bboxes(create_test_academic_pdf())
    titles = [section["title"].lower() for section in sections]
    assert any("abstract" in title for title in titles)
    assert any("introduction" in title for title in titles)
    assert any("results" in title for title in titles)
    assert any("discussion" in title for title in titles)
    assert all("conditions" not in title for title in titles)
    assert all("0.0059" not in title for title in titles)
    assert all("references" not in title for title in titles)
    combined = " ".join(section["clean_text"] for section in sections)
    assert "Tumor-specific" in combined
    # Base-14 PDF fonts replace Greek glyphs, but the marker line must survive.
    assert "IFN-" in combined and "TNF-" in combined


def test_rectangle_selection_uses_both_axes() -> None:
    pdf = create_test_academic_pdf()
    page = extract_page_blocks_for_selection(pdf, 1)
    left_region = {"x0": 0.05, "y0": 0.24, "x1": 0.50, "y1": 0.50}
    blocks = select_blocks_in_region(page["blocks"], page["width"], page["height"], left_region)
    selected_text = " ".join(block["text"] for block in blocks)
    assert "Single-cell" in selected_text
    assert "right column" not in selected_text


def test_forced_two_column_order_reads_left_before_right() -> None:
    blocks = [
        {
            "lines": [
                {"text": "Left line one.", "bbox": (50, 100, 250, 112)},
                {"text": "Right line one.", "bbox": (350, 100, 550, 112)},
                {"text": "Left line two.", "bbox": (50, 120, 250, 132)},
                {"text": "Right line two.", "bbox": (350, 120, 550, 132)},
            ]
        }
    ]
    region = {"x0": 0.05, "y0": 0.10, "x1": 0.95, "y1": 0.25}
    text = order_selected_text(blocks, 600, 800, region, layout_mode="two_columns")
    assert text.splitlines() == [
        "Left line one.",
        "Left line two.",
        "",
        "Right line one.",
        "Right line two.",
    ]


def test_pdf_span_spacing_is_reconstructed_from_geometry() -> None:
    spans = [
        {"text": "death", "bbox": (0, 0, 25, 10), "size": 10},
        {"text": "varies", "bbox": (27, 0, 52, 10), "size": 10},
        {"text": ",", "bbox": (52, 0, 54, 10), "size": 10},
        {"text": "micro", "bbox": (58, 0, 80, 10), "size": 10},
        {"text": "environment", "bbox": (80, 0, 125, 10), "size": 10},
    ]
    assert join_line_spans(spans) == "death varies, microenvironment"


def test_raised_numeric_citation_spans_can_be_removed() -> None:
    spans = [
        {
            "text": "Immune cells",
            "bbox": (0, 90, 58, 101),
            "origin": (0, 100),
            "size": 10,
            "flags": 0,
        },
        {
            "text": "1,2–4",
            "bbox": (58, 85, 75, 93),
            "origin": (58, 93),
            "size": 6,
            "flags": 1,
        },
        {
            "text": ".",
            "bbox": (75, 90, 78, 101),
            "origin": (75, 100),
            "size": 10,
            "flags": 0,
        },
    ]
    assert "1,2–4" in join_line_spans(spans)
    assert join_line_spans(spans, filter_superscript_citations=True) == "Immune cells."


def test_raised_citation_geometry_works_without_pdf_flag() -> None:
    spans = [
        {"text": "reported.", "bbox": (0, 90, 42, 101), "origin": (0, 100), "size": 10},
        {"text": "12", "bbox": (42, 84, 50, 92), "origin": (42, 92), "size": 6},
        {"text": " Next", "bbox": (52, 90, 78, 101), "origin": (52, 100), "size": 10},
    ]
    assert join_line_spans(spans, filter_superscript_citations=True) == "reported. Next"


def test_raised_citation_at_end_of_line_is_removed() -> None:
    spans = [
        {
            "text": "immune responses.",
            "bbox": (0, 90, 75, 101),
            "origin": (0, 100),
            "size": 10,
            "flags": 0,
        },
        {
            "text": "37,40,41",
            "bbox": (75, 84, 103, 92),
            "origin": (75, 92),
            "size": 6,
            "flags": 1,
        },
    ]
    assert join_line_spans(spans, filter_superscript_citations=True) == "immune responses."


@pytest.mark.parametrize(
    ("base", "superscript", "suffix"),
    [
        ("10", "6", " cells"),
        ("m", "2", " area"),
        ("x", "2", " value"),
        ("Ca", "2+", " channels"),
    ],
)
def test_scientific_superscript_spans_are_preserved(
    base: str, superscript: str, suffix: str
) -> None:
    spans = [
        {"text": base, "bbox": (0, 90, 12, 101), "origin": (0, 100), "size": 10},
        {
            "text": superscript,
            "bbox": (12, 84, 20, 92),
            "origin": (12, 92),
            "size": 6,
            "flags": 1,
        },
        {"text": suffix, "bbox": (22, 90, 60, 101), "origin": (22, 100), "size": 10},
    ]
    cleaned = join_line_spans(spans, filter_superscript_citations=True)
    assert superscript in cleaned


def test_selection_uses_superscript_cleaned_line_only_when_enabled() -> None:
    blocks = [
        {
            "lines": [
                {
                    "text": "Immune cells1,2.",
                    "text_without_superscript_citations": "Immune cells.",
                    "bbox": (50, 100, 250, 112),
                }
            ]
        }
    ]
    region = {"x0": 0.05, "y0": 0.10, "x1": 0.50, "y1": 0.20}
    original = order_selected_text(blocks, 600, 800, region, layout_mode="single_column")
    filtered = order_selected_text(
        blocks,
        600,
        800,
        region,
        layout_mode="single_column",
        filter_superscript_citations=True,
    )
    assert original == "Immune cells1,2."
    assert filtered == "Immune cells."


def test_unicode_superscript_citations_preserve_scientific_notation() -> None:
    raw = "Immune cells¹,²–⁴ remained at 10⁶ cells per m² with Ca²⁺ channels and x² values."
    cleaned = remove_unicode_superscript_citations(raw)
    assert "cells¹" not in cleaned
    assert "10⁶" in cleaned
    assert "m²" in cleaned
    assert "Ca²⁺" in cleaned
    assert "x²" in cleaned
    assert "cells¹" in clean_academic_text(raw, filter_superscript_citations=False)


def test_missing_spaces_inside_one_pdf_span_are_rebuilt_from_glyphs() -> None:
    characters = []
    cursor = 0.0
    for word_index, word in enumerate(["death", "varies", "greatly"]):
        if word_index:
            cursor += 2.4  # Visible word gap, but deliberately no space glyph.
        for character in word:
            characters.append({"c": character, "bbox": (cursor, 0, cursor + 4, 10)})
            cursor += 4
    spans = [{"size": 10, "chars": characters}]
    assert join_line_spans(spans) == "death varies greatly"


def test_sentence_spacing_is_restored_after_citation_removal() -> None:
    raw = "Individual tumors (Smith et al., 2020).Which subsets respond?Next sentence."
    assert clean_academic_text(raw) == "Individual tumors. Which subsets respond? Next sentence."


CELL_FIRST_PDF = Path(__file__).parent.parent / "references" / "CellFirst.pdf"


@pytest.mark.skipif(not CELL_FIRST_PDF.exists(), reason="CellFirst.pdf is not available")
def test_cellfirst_first_introduction_paragraph_has_real_word_spaces() -> None:
    pdf = CELL_FIRST_PDF.read_bytes()
    page = extract_page_blocks_for_selection(pdf, 2)
    region = {
        "x0": 50 / page["width"],
        "y0": 650 / page["height"],
        "x1": 305 / page["width"],
        "y1": 740 / page["height"],
    }
    blocks = select_blocks_in_region(page["blocks"], page["width"], page["height"], region)
    raw = order_selected_text(
        blocks,
        page["width"],
        page["height"],
        region,
        layout_mode="single_column",
    )
    cleaned = clean_academic_text(raw)

    assert "death vary greatly between different cancers and individual tumors." in cleaned
    assert "Which of the numerous cell subsets in a tumor contribute to the response" in cleaned
    assert "how their interactions are regulated" in cleaned
    assert "deathvary" not in cleaned
    assert "numerouscellsubsets" not in cleaned
    assert "atumorcontribute" not in cleaned


def test_page_rendering_returns_png() -> None:
    pdf = create_test_academic_pdf()
    sections = extract_sections_with_bboxes(pdf)
    bboxes = sections[0]["bboxes_by_page"].get(1, [])
    rendered = render_page_with_red_underlines(pdf, page_num=1, bboxes=bboxes, dpi=120)
    assert rendered.startswith(b"\x89PNG")


def test_tts_chunking_is_bounded_and_lossless() -> None:
    text = " ".join(f"Sentence {index} contains several useful words." for index in range(100))
    chunks = chunk_text(text, max_chars=240)
    assert len(chunks) > 1
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert " ".join(chunks) == text


def test_default_tts_chunking_splits_a_full_paragraph() -> None:
    text = " ".join(
        f"Synthetic sentence {index} contains enough ordinary words for testing."
        for index in range(30)
    )
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(chunk) <= DEFAULT_CHUNK_CHARS for chunk in chunks)
    assert " ".join(chunks) == text


def test_edge_cbr_mp3_duration_estimate() -> None:
    # MPEG-2 Layer III, bitrate index 6 = 48 kbps.
    audio = bytes.fromhex("fff364c4") + bytes(5996)
    assert estimate_mp3_duration(audio) == pytest.approx(1.0, abs=0.01)


def test_normalized_block_boxes_are_clamped_to_the_unit_square() -> None:
    blocks = [
        {"bbox": (61.2, 79.2, 306.0, 396.0)},
        {"bbox": (-5.0, -5.0, 700.0, 900.0)},
    ]
    boxes = normalized_block_boxes(blocks, 612.0, 792.0)
    assert boxes[0] == pytest.approx({"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}, rel=1e-6)
    assert boxes[1] == {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}


def test_normalized_block_boxes_skip_unusable_geometry() -> None:
    assert normalized_block_boxes([{"bbox": (0, 0, 1, 1)}], 0.0, 792.0) == []
    assert normalized_block_boxes([{"text": "no bbox"}], 612.0, 792.0) == []
    assert normalized_block_boxes([{"bbox": ("a", 0, 1, 1)}], 612.0, 792.0) == []


def test_normalized_boxes_agree_with_region_selection() -> None:
    """The component highlights blocks from normalized boxes using the rule in
    select_blocks_in_region. Both must agree, or the preview lies."""
    page_width, page_height = 612.0, 792.0
    blocks = [
        {"bbox": (61.2, 79.2, 306.0, 158.4), "text": "inside"},
        {"bbox": (61.2, 633.6, 306.0, 712.8), "text": "far below"},
        {"bbox": (275.4, 79.2, 520.2, 158.4), "text": "straddling the edge"},
    ]
    region = {"x0": 0.05, "y0": 0.05, "x1": 0.5, "y1": 0.3}
    boxes = normalized_block_boxes(blocks, page_width, page_height)

    minimum_area = 1 / (page_width * page_height)
    highlighted = []
    for index, box in enumerate(boxes):
        overlap_x = max(0.0, min(region["x1"], box["x1"]) - max(region["x0"], box["x0"]))
        overlap_y = max(0.0, min(region["y1"], box["y1"]) - max(region["y0"], box["y0"]))
        area = max(minimum_area, (box["x1"] - box["x0"]) * (box["y1"] - box["y0"]))
        centre_x = (box["x0"] + box["x1"]) / 2
        centre_y = (box["y0"] + box["y1"]) / 2
        centre_inside = (
            region["x0"] <= centre_x <= region["x1"] and region["y0"] <= centre_y <= region["y1"]
        )
        if centre_inside or (overlap_x * overlap_y) / area >= 0.10:
            highlighted.append(index)

    selected = select_blocks_in_region(blocks, page_width, page_height, region)
    assert [blocks[index]["text"] for index in highlighted] == [block["text"] for block in selected]


def test_generate_speech_reports_progress_per_chunk(monkeypatch) -> None:
    import tts_engine

    async def fake_chunk(text: str, voice: str, rate: str, timeout_seconds: float) -> bytes:
        return b"\xff\xf3\x64\xc4" + bytes(300)

    monkeypatch.setattr(tts_engine, "edge_tts", object())
    monkeypatch.setattr(tts_engine, "_generate_chunk", fake_chunk)

    reported: list[tuple[int, int]] = []
    text = ". ".join(f"Sentence number {index} about tumor-specific cells" for index in range(60))
    audio = asyncio.run(
        generate_speech_async(text, progress=lambda done, total: reported.append((done, total)))
    )

    expected = len(chunk_text(text))
    assert expected >= 2
    assert reported == [(index + 1, expected) for index in range(expected)]
    assert len(audio) > 0


REAL_PDF = os.environ.get("IMMUNOLOGY_TEST_PDF")


@pytest.mark.skipif(not REAL_PDF, reason="Set IMMUNOLOGY_TEST_PDF for the local paper regression")
def test_real_immunology_paper_has_no_statistical_section_titles() -> None:
    pdf_path = Path(REAL_PDF)
    sections = extract_sections_with_bboxes(pdf_path.read_bytes())
    titles = [section["title"] for section in sections]
    assert sections
    assert not any("%" in title or "P =" in title or "Conditions" in title for title in titles)
    assert any("Results" in title for title in titles)
    assert any("Discussion" in title for title in titles)
    assert any("Method Details" in title for title in titles)
