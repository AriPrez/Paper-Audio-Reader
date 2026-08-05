"""Streamlit interface for the academic PDF audio reader."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import fitz
import streamlit as st

# Keep imports stable whether launched from this directory or from the repo root.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from canvas_select import render_crosshair_canvas_selector
from parser import (
    clean_academic_text,
    extract_page_blocks_for_selection,
    order_selected_text,
    render_page_with_red_underlines,
    select_blocks_in_region,
)
from tts_engine import DEFAULT_CHUNK_CHARS, VOICES, chunk_text, estimate_mp3_duration, generate_speech


st.set_page_config(
    page_title="Paper Audio Reader",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 98%; }
      .stButton > button { border-radius: 8px; font-weight: 650; }
      .reader-note { color: #58717a; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULTS = {
    "audio_cache": {},
    "pdf_bytes": None,
    "pdf_id": None,
    "pdf_page_num": 1,
    "pdf_zoom_dpi": 150,
    "crop_boxes": {},
    "selector_revision": 0,
}
for state_name, default in DEFAULTS.items():
    if state_name not in st.session_state:
        st.session_state[state_name] = default.copy() if isinstance(default, dict) else default


def reset_document(pdf_bytes: bytes) -> None:
    """Reset all document-dependent state after an upload."""
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pdf_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    st.session_state.pdf_page_num = 1
    st.session_state.pdf_zoom_dpi = 150
    st.session_state.crop_boxes = {}
    st.session_state.selector_revision += 1
    st.session_state.audio_cache.clear()
    st.cache_data.clear()


def cache_key(text: str, voice: str, speed: float) -> str:
    """Content-address audio cache key containing every audio input."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"audio-v2:{digest}:{voice}:{speed:.2f}"


def render_audio_panel(text: str, voice: str, speed: float, key_prefix: str) -> None:
    """Render generation controls and a cached audio player."""
    words = len(text.split())
    estimate = round(words / max(1.0, 150.0 * speed), 1) if words else 0
    speech_chunks = chunk_text(text, max_chars=DEFAULT_CHUNK_CHARS) if text else []
    st.caption(
        f"{words} words · approximately {estimate:g} min · "
        f"{len(speech_chunks)} speech segment(s) · Edge TTS requires Internet"
    )
    key = cache_key(text, voice, speed) if text else "empty"

    if st.button(
        "🔊 Generate and play audio",
        key=f"{key_prefix}_generate",
        use_container_width=True,
        type="primary",
        disabled=not bool(text.strip()),
    ):
        with st.spinner("Generating speech in bounded chunks…"):
            try:
                st.session_state.audio_cache[key] = generate_speech(
                    text,
                    voice=voice,
                    rate=speed,
                    timeout_seconds=35,
                    max_chars=DEFAULT_CHUNK_CHARS,
                )
            except Exception as exc:
                st.error(str(exc))

    if key in st.session_state.audio_cache:
        audio = st.session_state.audio_cache[key]
        duration = estimate_mp3_duration(audio)
        if duration is not None:
            minutes, seconds = divmod(round(duration), 60)
            st.success(f"Complete MP3 ready · {minutes}:{seconds:02d}")
        st.audio(audio, format="audio/mp3")
        st.download_button(
            "Download complete MP3",
            data=audio,
            file_name="paper-selection.mp3",
            mime="audio/mpeg",
            key=f"{key_prefix}_download_{key[:24]}",
            use_container_width=True,
        )


def render_page_controls(total_pages: int) -> None:
    """Render page navigation and raster quality controls."""
    previous, following, jump, status = st.columns([1, 1, 1.5, 1.7], gap="small")
    with previous:
        if st.button(
            "◀ Previous",
            use_container_width=True,
            disabled=st.session_state.pdf_page_num <= 1,
        ):
            st.session_state.pdf_page_num -= 1
            st.rerun()
    with following:
        if st.button(
            "Next ▶",
            use_container_width=True,
            disabled=st.session_state.pdf_page_num >= total_pages,
        ):
            st.session_state.pdf_page_num += 1
            st.rerun()
    with jump:
        requested_page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=int(st.session_state.pdf_page_num),
            step=1,
            label_visibility="collapsed",
        )
        if requested_page != st.session_state.pdf_page_num:
            st.session_state.pdf_page_num = int(requested_page)
            st.rerun()
    with status:
        st.markdown(f"**Page {st.session_state.pdf_page_num} / {total_pages}**")

    quality = st.radio(
        "Render quality",
        options=[120, 150, 200, 250],
        index=[120, 150, 200, 250].index(st.session_state.pdf_zoom_dpi)
        if st.session_state.pdf_zoom_dpi in [120, 150, 200, 250]
        else 1,
        format_func=lambda value: f"{value} dpi",
        horizontal=True,
    )
    st.session_state.pdf_zoom_dpi = quality


st.sidebar.title("📖 Paper Audio Reader")
st.sidebar.caption("Biomedical-safe PDF extraction and speech")
uploaded_file = st.sidebar.file_uploader("Upload a research paper", type=["pdf"])
if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    uploaded_id = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
    if uploaded_id != st.session_state.pdf_id:
        reset_document(uploaded_bytes)

if st.session_state.pdf_bytes is not None and st.sidebar.button("Close current PDF"):
    st.session_state.pdf_bytes = None
    st.session_state.pdf_id = None
    st.session_state.audio_cache.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Text layout")
layout_mode = st.sidebar.selectbox(
    "Reading order",
    options=["automatic", "single_column", "two_columns"],
    format_func=lambda value: {
        "automatic": "Automatic",
        "single_column": "Single column / rows",
        "two_columns": "Two columns: left, then right",
    }[value],
    help="For a two-column paragraph, the center of the rectangle is used as the column gutter.",
)

st.sidebar.subheader("Voice")
voice_label = st.sidebar.selectbox("Voice", list(VOICES), label_visibility="collapsed")
voice_name = VOICES[voice_label]
speed_rate = st.sidebar.slider("Speed", 0.8, 2.0, 1.0, 0.1)

st.sidebar.subheader("Cleaning")
filter_brackets = st.sidebar.checkbox("Bracket citations [1-5]", value=True)
filter_parentheses = st.sidebar.checkbox("Parenthetical citations (Smith 2020 / (1-3))", value=True)
filter_superscript_citations = st.sidebar.checkbox(
    "Superscript numeric citations ¹–⁵",
    value=True,
    help="Uses PDF font size and baseline position while preserving scientific exponents such as 10⁶, m² and Ca²⁺.",
)
filter_urls = st.sidebar.checkbox("URLs and DOIs", value=True)
filter_captions = st.sidebar.checkbox("Figures, captions and tables", value=True)
filter_equations = st.sidebar.checkbox("Standalone equations", value=True)


if st.session_state.pdf_bytes is None:
    st.info("Upload a scientific PDF from the sidebar to begin.")
    st.markdown(
        """
        The reader preserves biomedical notation, removes optional citation noise,
        and reads only the text inside the rectangle you draw on a page.

        PDF extraction and viewing are local. Speech uses the free online Edge TTS
        service and therefore requires an Internet connection.
        """
    )
    st.stop()


try:
    document = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
    if document.needs_pass:
        document.close()
        st.error("This PDF is password-protected and cannot be opened.")
        st.stop()
    total_pdf_pages = len(document)
    text_sample = "".join(document[index].get_text().strip() for index in range(min(3, len(document))))
    document.close()
except Exception as exc:
    st.error(f"Invalid or unsupported PDF: {exc}")
    st.stop()

if not total_pdf_pages:
    st.error("The PDF contains no pages.")
    st.stop()
if not text_sample:
    st.warning("No selectable text was found. This appears to be a scanned PDF; OCR is required first.")

st.session_state.pdf_page_num = min(max(1, st.session_state.pdf_page_num), total_pdf_pages)
left_column, right_column = st.columns([1.18, 1.0], gap="large")

with left_column:
    st.subheader("🎯 Rectangle selection")
    render_page_controls(total_pdf_pages)
    active_page = st.session_state.pdf_page_num
    try:
        rendered = render_page_with_red_underlines(
            st.session_state.pdf_bytes,
            page_num=active_page,
            bboxes=[],
            dpi=st.session_state.pdf_zoom_dpi,
        )
        initial_selection = st.session_state.crop_boxes.get(active_page)
        selection = render_crosshair_canvas_selector(
            rendered,
            width=760,
            initial=initial_selection,
            key=(
                f"selector_{st.session_state.pdf_id}_{active_page}_"
                f"{st.session_state.selector_revision}"
            ),
        )
        if selection:
            st.session_state.crop_boxes[active_page] = selection
        if st.button("Clear selection", disabled=not bool(selection or initial_selection)):
            st.session_state.crop_boxes.pop(active_page, None)
            st.session_state.selector_revision += 1
            st.rerun()
    except Exception as exc:
        selection = None
        st.error(f"Interactive page rendering failed: {exc}")

effective_selection = selection or st.session_state.crop_boxes.get(st.session_state.pdf_page_num)
try:
    page_data = extract_page_blocks_for_selection(
        st.session_state.pdf_bytes,
        st.session_state.pdf_page_num,
    )
    selected_blocks = select_blocks_in_region(
        page_data["blocks"],
        page_data["width"],
        page_data["height"],
        effective_selection,
    )
    raw_selection = order_selected_text(
        selected_blocks,
        page_data["width"],
        page_data["height"],
        effective_selection,
        layout_mode=layout_mode,
        filter_superscript_citations=filter_superscript_citations,
    )
    clean_selection = clean_academic_text(
        raw_selection,
        filter_brackets=filter_brackets,
        filter_parentheses=filter_parentheses,
        filter_superscript_citations=filter_superscript_citations,
        filter_urls=filter_urls,
        filter_captions=filter_captions,
        filter_equations_flag=filter_equations,
    )
except Exception as exc:
    selected_blocks = []
    clean_selection = ""
    st.error(f"Text selection failed: {exc}")

with right_column:
    st.subheader("🎧 Selected text")
    if not effective_selection:
        st.info("Drag a rectangle over one or more paragraphs on the page.")
    elif not selected_blocks:
        st.warning("The rectangle does not intersect selectable text.")
    else:
        layout_description = {
            "automatic": "automatic order",
            "single_column": "row-by-row order",
            "two_columns": "left column, then right column",
        }[layout_mode]
        st.caption(f"{len(selected_blocks)} PDF text block(s) selected · {layout_description}")
    render_audio_panel(clean_selection, voice_name, speed_rate, "rectangle")
    st.divider()
    st.text_area("Clean selected transcript", clean_selection, height=420, disabled=True)
