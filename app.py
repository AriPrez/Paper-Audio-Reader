"""Streamlit interface for the academic PDF audio reader."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import threading

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
from audio_queue import render_audio_queue
from tts_engine import (
    DEFAULT_CHUNK_CHARS,
    VOICES,
    chunk_text,
    estimate_mp3_duration,
    join_mp3_parts,
    plan_parts,
    project_total_seconds,
    render_part,
)


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
    "started": False,
    "audio_stream": None,
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


@st.cache_resource(show_spinner=False)
def warm_speech_service() -> bool:
    """Pay Edge's one-time setup before anybody is waiting on it.

    The first synthesis in a process is far slower than the rest: measured in a
    fresh interpreter, 14.1s for the first request against 0.7-3.9s for the
    next four. Whoever generates first on a freshly started server pays that,
    and it is most of the difference between a two-second wait and a ten-second
    one. Warming it when a document is opened moves the cost to a moment when
    nobody has asked for anything yet.

    A fixed word is sent, never the document: see the privacy note. It runs on
    a daemon thread so opening a PDF is not held up, and failure is ignored
    because this is an optimisation, not a step.
    """

    def ping() -> None:
        try:
            render_part(["Ready."], timeout_seconds=20)
        except Exception:
            pass

    threading.Thread(target=ping, daemon=True).start()
    return True


def cache_key(text: str, voice: str, speed: float) -> str:
    """Content-address audio cache key containing every audio input."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"audio-v2:{digest}:{voice}:{speed:.2f}"


# Streamlit publishes no theme CSS variable and no data-theme attribute in the
# DOM, so an accent used inside inline SVG cannot be expressed in CSS alone. It
# is resolved here instead; switching theme reruns the script, which redraws it.
ACCENTS = {"light": "#0F7C91", "dark": "#4FBFD6"}


def theme_accent() -> str:
    try:
        return ACCENTS.get(st.context.theme.type or "light", ACCENTS["light"])
    except Exception:  # No script context: tests and bare imports.
        return ACCENTS["light"]


def hero_mark(accent: str) -> str:
    """Draw the product rather than a symbol for it.

    A document-with-soundwaves icon says "audio" and nothing more. What makes
    this tool what it is, is the gesture: a rectangle over three lines of a
    page, and only those lines coming out as speech. So the mark shows the
    selection, the captured lines in the accent colour, and the sound leaving
    the page. The greys are currentColor, so the paper follows the theme.
    """
    return f"""<svg viewBox="0 0 206 152" width="206" height="152" fill="none" aria-hidden="true">
<g stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" opacity=".34">
<path d="M24 11h72l24 24v98a4 4 0 0 1-4 4H24a4 4 0 0 1-4-4V15a4 4 0 0 1 4-4z"/>
<path d="M96 11v20a4 4 0 0 0 4 4h20"/>
</g>
<g stroke="currentColor" stroke-width="3" stroke-linecap="round" opacity=".19">
<path d="M33 52h54M33 62h66"/>
<path d="M33 114h66M33 124h38"/>
</g>
<g stroke="{accent}" stroke-width="3" stroke-linecap="round" opacity=".85">
<path d="M33 78h66M33 89h62M33 100h48"/>
</g>
<rect x="27" y="70" width="80" height="38" rx="2"
      fill="{accent}" fill-opacity=".08" stroke="{accent}" stroke-width="1.5"/>
<g fill="{accent}">
<rect x="24" y="67" width="6" height="6" rx="1"/>
<rect x="104" y="67" width="6" height="6" rx="1"/>
<rect x="24" y="105" width="6" height="6" rx="1"/>
<rect x="104" y="105" width="6" height="6" rx="1"/>
</g>
<g stroke="{accent}" stroke-width="2" stroke-linecap="round">
<path d="M142 77a14 14 0 0 1 0 24" opacity=".9"/>
<path d="M156 67a24 24 0 0 1 0 44" opacity=".62"/>
<path d="M170 57a34 34 0 0 1 0 64" opacity=".34"/>
</g>
</svg>"""


STEPS = (
    ("1", "Upload", "A PDF with a text layer. Scanned pages need OCR first."),
    ("2", "Select", "Drag a rectangle on the page, or click a paragraph to take it whole."),
    ("3", "Listen", "Check the transcript, edit it if needed, then generate an MP3."),
)

PRIVACY_NOTE = (
    "PDF parsing and page rendering run locally in this Streamlit process. "
    "Speech is produced by the online Microsoft Edge TTS service, so the cleaned "
    "text of your selection leaves the machine when you press generate. "
    "Avoid confidential, identifiable clinical or unpublished content."
)

# Only the entry screens. The sidebar is every setting of an application that
# has not been opened yet, which reads as a preferences panel and drowns the one
# action on the page; the wide layout exists for the reader, not for a page with
# a single button on it.
# Selectors are scoped under the markdown container on purpose. Streamlit's own
# emotion rules for p/h1 inside it are more specific than a bare class, and win:
# an unscoped `.note { margin-top }` computes to 0px.
ENTRY_CSS = """
<style>
  section[data-testid="stSidebar"],
  div[data-testid="stSidebarCollapsedControl"],
  div[data-testid="stAppDeployButton"] { display: none !important; }
  .stApp .block-container { max-width: 52rem; padding-top: 3.2rem; }

  div[data-testid="stMarkdownContainer"] .entry { text-align: center; }
  div[data-testid="stMarkdownContainer"] .entry .eyebrow {
    text-transform: uppercase; letter-spacing: .15em;
    font-size: .72rem; font-weight: 600; opacity: .5; margin: 0 0 1.1rem;
  }
  div[data-testid="stMarkdownContainer"] .entry svg { width: 246px; height: auto; }
  /* Streamlit rewrites the h1 into a heading widget and injects an anchor-link
     span inside it. Outside its usual layout that span goes into the flow and
     adds ~250px of blank height under the title. A landing page has no section
     to link to anyway. */
  div[data-testid="stMarkdownContainer"] .entry [data-testid="stHeaderActionElements"] {
    display: none;
  }
  div[data-testid="stMarkdownContainer"] .entry h1 {
    font-size: 2.7rem; line-height: 1.12; letter-spacing: -0.028em;
    padding: 0; margin: .9rem 0 .8rem;
  }
  div[data-testid="stMarkdownContainer"] .entry .lede {
    max-width: 33rem; margin: 0 auto; opacity: .7;
    font-size: 1.08rem; line-height: 1.6;
  }
  div[data-testid="stMarkdownContainer"] .entry.compact h1 {
    font-size: 1.95rem; margin: 0 0 .5rem;
  }
  div[data-testid="stMarkdownContainer"] .entry.compact .lede { font-size: 1rem; }

  /* A sequence, not three interchangeable boxes: the numeral carries the order
     so the labels do not have to repeat it. They are divs rather than headings
     because Streamlit rewrites any h1-h6 in this HTML into a heading widget
     with its own anchor link — these are labels, not document sections. */
  div[data-testid="stMarkdownContainer"] .steps {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;
    margin: 3.6rem 0 0; text-align: left;
  }
  div[data-testid="stMarkdownContainer"] .step {
    border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent);
    padding-top: .85rem;
  }
  div[data-testid="stMarkdownContainer"] .step .n {
    font-size: .95rem; font-weight: 700; opacity: .3;
  }
  div[data-testid="stMarkdownContainer"] .step .t {
    font-size: 1rem; font-weight: 600; margin: .1rem 0 .35rem;
  }
  div[data-testid="stMarkdownContainer"] .step p {
    font-size: .88rem; line-height: 1.55; opacity: .62; margin: 0;
  }
  @media (max-width: 640px) {
    div[data-testid="stMarkdownContainer"] .steps { grid-template-columns: 1fr; gap: 1.3rem; }
  }

  div[data-testid="stMarkdownContainer"] .note {
    margin: 3.2rem auto 0; max-width: 38rem;
    font-size: .82rem; line-height: 1.6; opacity: .5; text-align: center;
  }
  div[data-testid="stMarkdownContainer"] .note strong { opacity: .9; }
</style>
"""


def accept_upload(container) -> None:
    """Render the single file uploader and load whatever it returns."""
    uploaded = container.file_uploader(
        "Research paper (PDF)",
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="collapsed" if container is not st.sidebar else "visible",
    )
    if uploaded is None:
        return
    uploaded_bytes = uploaded.getvalue()
    uploaded_id = hashlib.sha256(uploaded_bytes).hexdigest()[:16]
    if uploaded_id != st.session_state.pdf_id:
        reset_document(uploaded_bytes, uploaded.name)
        st.rerun()


def render_steps_and_privacy() -> None:
    # Written flush left on purpose, here and below: Markdown turns any line
    # indented by four spaces into a code block, which would render the tags as
    # literal text on the page.
    steps = "".join(
        f"""<div class="step"><div class="n">{number}</div>
<div class="t">{title}</div><p>{body}</p></div>"""
        for number, title, body in STEPS
    )
    st.markdown(
        f"""<div class="steps">{steps}</div>
<p class="note"><strong>Where your document goes.</strong> {PRIVACY_NOTE}</p>""",
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    """Entry screen: one thing to read, one thing to press."""
    st.markdown(ENTRY_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""<div class="entry">
<p class="eyebrow">Scientific PDF → speech</p>
{hero_mark(theme_accent())}
<h1>Read a scientific paper aloud</h1>
<p class="lede">Draw a rectangle over the paragraphs you want. Only those are
extracted, stripped of citation noise, and spoken — with the biomedical
notation pronounced properly.</p>
</div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    _, middle, _ = st.columns([1, 1, 1])
    with middle:
        if st.button(
            "Start reading",
            icon=":material/arrow_forward:",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.started = True
            st.rerun()
    render_steps_and_privacy()


def render_upload_screen() -> None:
    """Shown once past the landing, while no document is open."""
    st.markdown(ENTRY_CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="entry compact">
<h1>Choose a paper</h1>
<p class="lede">A PDF with a text layer. It stays on this machine.</p>
</div>""",
        unsafe_allow_html=True,
    )
    _, middle, _ = st.columns([1, 3, 1])
    with middle:
        accept_upload(middle)
    render_steps_and_privacy()


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
        st.session_state.audio_stream = {
            "job": key,
            "chunks": speech_chunks,
            "plan": plan_parts(len(speech_chunks)),
            "parts": {},
            "held": set(),
            "voice": voice,
            "speed": speed,
            "error": None,
        }
        st.rerun()

    stream = st.session_state.audio_stream
    streaming = isinstance(stream, dict) and stream.get("job") == key
    if streaming:
        plan = stream["plan"]
        # The player's place is claimed before the part is requested and filled
        # afterwards, so a finished part reaches it inside the same script run.
        # Rendering the player first instead would leave it empty until the
        # next rerun, putting a whole rerun — which re-sends the page image to
        # the selector — between the audio existing and anyone hearing it.
        player_slot = st.container()
        progress_slot = st.empty()

        finished = len(stream["parts"])
        if not stream["error"] and finished < len(plan):
            start, stop = plan[finished]
            # Progress is shown only while there is nothing to listen to yet.
            # Once the first part is playable the rest is none of the
            # listener's business, and a bar that keeps filling next to a
            # playing recording reads as a problem rather than as progress.
            bar = (
                progress_slot.progress(0.0, text="Generating the first part…")
                if finished == 0
                else None
            )
            try:
                stream["parts"][finished] = render_part(
                    stream["chunks"][start:stop],
                    voice=stream["voice"],
                    rate=stream["speed"],
                    timeout_seconds=45,
                    progress=(
                        (lambda made, total: bar.progress(made / total,
                            text=f"Generating the first part — segment {made} of {total}…"))
                        if bar is not None
                        else None
                    ),
                )
            except Exception as exc:
                # Recorded rather than raised: reruns drive this loop, and an
                # exception here would leave it retrying the same failed part.
                stream["error"] = str(exc)
            finally:
                progress_slot.empty()
            finished = len(stream["parts"])

        complete = finished >= len(plan)
        # The player reports what it actually holds; anything else is resent.
        # Tracking what was sent instead loses a part whenever a rerun replaces
        # the component's arguments before its iframe has finished mounting.
        pending = [
            (index, stream["parts"][index])
            for index in sorted(stream["parts"])
            if index not in stream["held"]
        ]
        with player_slot:
            made = [stream["parts"][index] for index in sorted(stream["parts"])]
            characters_done = sum(
                len(chunk) for chunk in stream["chunks"][: plan[finished - 1][1]]
            ) if finished else 0
            stream["held"] = render_audio_queue(
                job=stream["job"],
                new_parts=pending,
                total_parts=len(plan),
                done=complete,
                estimated_seconds=project_total_seconds(
                    made,
                    characters_done,
                    sum(len(chunk) for chunk in stream["chunks"]),
                    fallback=estimate * 60.0,
                ),
                key=f"{key_prefix}_queue",
            )

        if stream["error"]:
            st.error(stream["error"])
        elif not complete:
            st.rerun()
        elif key not in st.session_state.audio_cache:
            st.session_state.audio_cache[key] = join_mp3_parts(
                [stream["parts"][index] for index in sorted(stream["parts"])]
            )

    if key in st.session_state.audio_cache:
        audio = st.session_state.audio_cache[key]
        duration = estimate_mp3_duration(audio)
        if duration is not None:
            minutes, seconds = divmod(round(duration), 60)
            st.caption(f"Complete MP3 ready · {minutes}:{seconds:02d}")
        # Only one player at a time: the queue player above may be mid-playback,
        # and a second element would talk over it.
        if not streaming:
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

# One uploader, moved rather than duplicated: it is the whole point of the
# screen while no document is open, and a sidebar detail once one is.
document_open = st.session_state.pdf_bytes is not None
if document_open:
    accept_upload(st.sidebar)

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


if not st.session_state.started:
    render_landing()
    st.stop()

if st.session_state.pdf_bytes is None:
    render_upload_screen()
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

# A document is open, so speech is likely and the wait for it starts now, in
# the background, rather than when the first selection is ready to be read.
warm_speech_service()

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
