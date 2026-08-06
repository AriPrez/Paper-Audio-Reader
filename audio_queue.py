"""Player that accepts audio parts as they are generated and chains them."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "paper_audio_queue_player",
    path=Path(__file__).parent / "audio_queue_component",
)


def render_audio_queue(
    job: str,
    new_parts: list[tuple[int, bytes]],
    total_parts: int | None,
    done: bool,
    estimated_seconds: float = 0.0,
    key: str = "audio_queue",
) -> set[int]:
    """Hand newly finished parts to the player without interrupting playback.

    Only parts the component has not received yet are passed in, and it is the
    component that says which those are — the returned set. Re-sending them all
    on every render would put megabytes of base64 on the websocket whenever
    anything else on the page reruns, and assuming a send arrived is worse: a
    rerun can replace an element's arguments before a freshly mounted iframe
    has received the previous ones, and that part is lost for good. Asking is
    also self-healing, since a remounted iframe reports an empty inventory.

    ``job`` identifies the recording. When it changes the component drops what
    it holds, otherwise a new selection would play after the previous one.

    ``estimated_seconds`` scales the progress bar before the real length is
    known, so the thumb does not jump backwards each time a part arrives.
    """
    value = _COMPONENT(
        job=job,
        parts=[
            {"index": index, "data": base64.b64encode(audio).decode("ascii")}
            for index, audio in new_parts
        ],
        total_parts=int(total_parts) if total_parts else None,
        done=bool(done),
        estimated_seconds=float(estimated_seconds or 0.0),
        default=None,
        key=key,
    )
    if not isinstance(value, dict) or value.get("job") != job:
        # No inventory yet, or one belonging to the previous recording.
        return set()
    try:
        return {int(index) for index in value.get("have") or []}
    except (TypeError, ValueError):
        return set()
