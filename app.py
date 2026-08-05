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
    expand_for_speech,
    extract_page_blocks_for_selection,
    normalized_block_boxes,
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
    menu_items={
        "Get Help": "https://github.com/AriPrez/Paper-Audio-Reader#readme",
        "Report a bug": "https://github.com/AriPrez/Paper-Audio-Reader/issues",
        "About": (
            "**Paper Audio Reader** — read scientific PDFs aloud without citation noise.\n\n"
            "PDF parsing and rendering are local; speech uses the online Microsoft Edge TTS "
            "service. MIT licensed."
        ),
    },
)

# Layout and typography only. Every colour comes from .streamlit/config.toml so
# that light and dark modes stay correct without a second stylesheet.
st.markdown(
    """
    <style>
      .block-container { padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1600px; }
      section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      h3 { margin-bottom: .35rem; }
      .stButton > button { font-weight: 600; }
      .stTextArea textarea { line-height: 1.62; font-size: .95rem; }
      .stAudio { width: 100%; }
      div[data-testid="stVerticalBlockBorderWrapper"] { padding-top: .15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


RENDER_QUALITIES = [120, 150, 200, 250]

DEFAULTS = {
    "audio_cache": {},
    "pdf_bytes": None,
    "pdf_id": None,
    "pdf_name": "",
    "pdf_page_num": 1,
    "pdf_zoom_dpi": 150,
    "crop_boxes": {},
    "selector_revision": 0,
}
for state_name, default in DEFAULTS.items():
    if state_name not in st.session_state:
        st.session_state[state_name] = default.copy() if isinstance(default, dict) else default


def reset_document(pdf_bytes: bytes, name: str = "") -> None:
    """Reset all document-dependent state after an upload."""
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pdf_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    st.session_state.pdf_name = name
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


def render_empty_state() -> None:
    """Landing screen shown until a document is uploaded."""
    st.header("Read a scientific paper aloud")
    st.markdown(
        "Listen to a paper without the citation noise. "
        "Draw a rectangle over the paragraphs you want, and only those are extracted, "
        "cleaned and spoken."
    )
    st.write("")

    first, second, third = st.columns(3, gap="medium")
    steps = (
        (first, "1 · Upload", "Load a PDF with a text layer from the sidebar. Scanned pages need OCR first."),
        (second, "2 · Select", "Drag a rectangle on the page. Two-column paragraphs are read left column first."),
        (third, "3 · Listen", "Check the transcript, edit it if needed, then generate an MP3 you can download."),
    )
    for column, heading, body in steps:
        with column, st.container(border=True):
            st.markdown(f"**{heading}**")
            st.caption(body)

    st.write("")
    with st.container(border=True):
        st.markdown("**Where your document goes**")
        st.caption(
            "PDF parsing and page rendering run locally in this Streamlit process. "
            "Speech is produced by the online Microsoft Edge TTS service, so the cleaned "
            "text of your selection leaves the machine when you press generate. "
            "Avoid confidential, identifiable clinical or unpublished content."
        )


def render_audio_panel(
    text: str,
    voice: str,
    speed: float,
    key_prefix: str,
    expand_notation: bool = True,
) -> None:
    """Render generation controls and a cached audio player.

    The engine receives the expanded notation while the transcript above stays
    readable, so what is spoken is shown separately rather than substituted.
    """
    spoken = expand_for_speech(text) if expand_notation else text
    words = len(spoken.split())
    estimate = round(words / max(1.0, 150.0 * speed), 1) if words else 0
    speech_chunks = chunk_text(spoken, max_chars=DEFAULT_CHUNK_CHARS) if spoken else []
    st.caption(
        f"{words} words · approximately {estimate:g} min · "
        f"{len(speech_chunks)} speech segment(s) · Edge TTS requires Internet"
    )
    if spoken and spoken != text:
        with st.expander("What the voice will read"):
            st.write(spoken)
    key = cache_key(spoken, voice, speed) if spoken else "empty"

    if st.button(
        "Generate and play audio",
        icon=":material/graphic_eq:",
        key=f"{key_prefix}_generate",
        use_container_width=True,
        type="primary",
        disabled=not bool(spoken.strip()),
    ):
        # The bar sits at zero until the first segment lands, so say what is
        # actually happening rather than "contacting": every segment is already
        # in flight, and the wait is the service's, not a connection being set
        # up. Naming the count also makes a long selection self-explanatory.
        total_segments = len(speech_chunks)
        progress_slot = st.empty()
        bar = progress_slot.progress(
            0.0,
            text=(
                f"Generating {total_segments} segment{'s' if total_segments > 1 else ''}"
                f"{' (4 at a time)' if total_segments > 1 else ''}…"
            ),
        )
        try:
            st.session_state.audio_cache[key] = generate_speech(
                spoken,
                voice=voice,
                rate=speed,
                timeout_seconds=45,
                max_chars=DEFAULT_CHUNK_CHARS,
                progress=lambda done, total: bar.progress(
                    done / total, text=f"Segment {done} of {total} generated"
                ),
            )
        except Exception as exc:
            st.error(str(exc))
        finally:
            progress_slot.empty()

    if key in st.session_state.audio_cache:
        audio = st.session_state.audio_cache[key]
        duration = estimate_mp3_duration(audio)
        if duration is not None:
            minutes, seconds = divmod(round(duration), 60)
            st.caption(f"Complete MP3 ready · {minutes}:{seconds:02d}")
        st.audio(audio, format="audio/mp3")
        st.download_button(
            "Download MP3",
            icon=":material/download:",
            data=audio,
            file_name="paper-selection.mp3",
            mime="audio/mpeg",
            key=f"{key_prefix}_download_{key[:24]}",
            use_container_width=True,
        )


def render_page_controls(total_pages: int) -> None:
    """Render page navigation and raster quality controls."""
    previous, following, jump, status = st.columns(
        [1.5, 1.2, 1.3, 3.4], gap="small", vertical_alignment="center"
    )
    # Labelled rather than icon-only: an icon with a tooltip leaves assistive
    # technology with nothing to announce.
    with previous:
        if st.button(
            "Previous",
            icon=":material/chevron_left:",
            use_container_width=True,
            disabled=st.session_state.pdf_page_num <= 1,
        ):
            st.session_state.pdf_page_num -= 1
            st.rerun()
    with following:
        if st.button(
            "Next",
            icon=":material/chevron_right:",
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
        st.caption(f"of {total_pages}")


st.sidebar.title("Paper Audio Reader")
st.sidebar.caption("Biomedical-safe PDF extraction and speech")
uploaded_file = st.sidebar.file_uploader("Research paper (PDF)", type=["pdf"])
if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    uploaded_id = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
    if uploaded_id != st.session_state.pdf_id:
        reset_document(uploaded_bytes, uploaded_file.name)

if st.session_state.pdf_bytes is not None and st.sidebar.button(
    "Close current PDF", icon=":material/close:", use_container_width=True
):
    st.session_state.pdf_bytes = None
    st.session_state.pdf_id = None
    st.session_state.pdf_name = ""
    st.session_state.audio_cache.clear()
    st.rerun()

st.sidebar.divider()

layout_mode = st.sidebar.selectbox(
    "Reading order",
    # Single column first: it is the default because it is the safe reading of
    # an arbitrary rectangle, and the automatic detector can reorder a
    # paragraph it misreads as two columns.
    options=["single_column", "automatic", "two_columns"],
    format_func=lambda value: {
        "automatic": "Automatic",
        "single_column": "Single column / rows",
        "two_columns": "Two columns: left, then right",
    }[value],
    help="For a two-column paragraph, the center of the rectangle is used as the column gutter.",
)

quality = st.sidebar.selectbox(
    "Render quality",
    options=RENDER_QUALITIES,
    index=RENDER_QUALITIES.index(st.session_state.pdf_zoom_dpi)
    if st.session_state.pdf_zoom_dpi in RENDER_QUALITIES
    else 1,
    format_func=lambda value: f"{value} dpi",
    help="Resolution of the page image. Higher is sharper to select on, and slower to render.",
)
st.session_state.pdf_zoom_dpi = quality

st.sidebar.divider()

voice_label = st.sidebar.selectbox("Voice", list(VOICES))
voice_name = VOICES[voice_label]
speed_rate = st.sidebar.slider("Speed", 0.8, 2.0, 1.0, 0.1)
expand_notation = st.sidebar.checkbox(
    "Biomedical pronunciation",
    value=True,
    help=(
        "Speech engines read notation literally. This says “M S I high” for MSI-H, "
        "“C D 8 positive” for CD8+, “ten to the power of 6” for 10⁶ and "
        "“and colleagues” for et al. The transcript on screen is unchanged."
    ),
)

with st.sidebar.expander("Text cleaning", expanded=False):
    st.caption("Everything below is removed from the spoken text.")
    filter_brackets = st.checkbox("Bracket citations [1-5]", value=True)
    filter_parentheses = st.checkbox("Parenthetical citations (Smith 2020)", value=True)
    filter_superscript_citations = st.checkbox(
        "Superscript numeric citations ¹–⁵",
        value=True,
        help="Uses PDF font size and baseline position while preserving scientific exponents such as 10⁶, m² and Ca²⁺.",
    )
    filter_urls = st.checkbox("URLs and DOIs", value=True)
    filter_captions = st.checkbox("Figures, captions and tables", value=True)
    filter_equations = st.checkbox("Standalone equations", value=True)

active_filters = sum(
    [
        filter_brackets,
        filter_parentheses,
        filter_superscript_citations,
        filter_urls,
        filter_captions,
        filter_equations,
    ]
)
st.sidebar.caption(f"{active_filters} of 6 cleaning filters active")


if st.session_state.pdf_bytes is None:
    render_empty_state()
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

st.caption(
    f"{st.session_state.pdf_name or 'Document'} · "
    f"{total_pdf_pages} page{'s' if total_pdf_pages > 1 else ''}"
)

active_page = st.session_state.pdf_page_num

# The page layout is needed before the selector is drawn: the component
# previews which paragraphs the rectangle captures, and selects one on a click.
try:
    page_data = extract_page_blocks_for_selection(st.session_state.pdf_bytes, active_page)
    block_boxes = normalized_block_boxes(
        page_data["blocks"], page_data["width"], page_data["height"]
    )
except Exception as exc:
    page_data = None
    block_boxes = []
    st.error(f"Page analysis failed: {exc}")

left_column, right_column = st.columns([1.18, 1.0], gap="large")

with left_column:
    st.subheader("Page selection")
    render_page_controls(total_pdf_pages)
    try:
        rendered = render_page_with_red_underlines(
            st.session_state.pdf_bytes,
            page_num=active_page,
            bboxes=[],
            dpi=st.session_state.pdf_zoom_dpi,
        )
        selection = render_crosshair_canvas_selector(
            rendered,
            initial=st.session_state.crop_boxes.get(active_page),
            blocks=block_boxes,
            page_width=page_data["width"] if page_data else 612.0,
            page_height=page_data["height"] if page_data else 792.0,
            key=(
                f"selector_{st.session_state.pdf_id}_{active_page}_"
                f"{st.session_state.selector_revision}"
            ),
        )
        # The component owns clearing, so a missing value means "cleared here",
        # not "not answered yet": the stored box has to follow.
        if selection:
            st.session_state.crop_boxes[active_page] = selection
        else:
            st.session_state.crop_boxes.pop(active_page, None)
        st.caption(
            "Drag to draw · click a paragraph to take it whole · drag the handles to adjust · "
            "ctrl + wheel to zoom · arrow keys to nudge · Escape to clear"
        )
    except Exception as exc:
        selection = None
        st.error(f"Interactive page rendering failed: {exc}")

effective_selection = selection
try:
    if page_data is None:
        raise RuntimeError("the page could not be analysed")
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
    st.subheader("Transcript")
    if not effective_selection:
        st.caption("Nothing selected yet.")
    elif not selected_blocks:
        st.caption("The rectangle does not intersect selectable text.")
    else:
        layout_description = {
            "automatic": "automatic order",
            "single_column": "row-by-row order",
            "two_columns": "left column, then right column",
        }[layout_mode]
        paragraphs = len(selected_blocks)
        st.caption(f"{paragraphs} paragraph{'s' if paragraphs > 1 else ''} · {layout_description}")

    # The transcript is editable: fixing an odd extraction before synthesis is
    # faster than redrawing the rectangle. The key is derived from the extracted
    # text so a new selection replaces the box instead of keeping stale edits.
    transcript_key = f"transcript_{hashlib.sha256(clean_selection.encode('utf-8')).hexdigest()[:16]}"
    spoken_text = st.text_area(
        "Text sent to the voice engine",
        value=clean_selection,
        height=360,
        key=transcript_key,
        label_visibility="collapsed",
        placeholder="The cleaned text of your selection appears here, and can be edited before synthesis.",
    )

    with st.container(border=True):
        render_audio_panel(
            spoken_text, voice_name, speed_rate, "rectangle", expand_notation=expand_notation
        )

st.divider()
st.caption(
    "Paper Audio Reader · MIT · "
    "[source](https://github.com/AriPrez/Paper-Audio-Reader) · "
    "PDF parsing and rendering are local; speech is sent to the online Edge TTS service."
)
