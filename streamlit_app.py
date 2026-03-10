"""
SoulX-FlashHead — Streamlit UI
================================
Run this AFTER the backend server is running:

    python server.py          # Terminal 1
    streamlit run streamlit_app.py  # Terminal 2
"""

import asyncio
import glob
import io
import json
import os
import subprocess
import tempfile
import time
import threading

import httpx
import imageio
import librosa
import numpy as np
import streamlit as st
import websockets

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SoulX-FlashHead",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
SAMPLE_RATE = 16000
TGT_FPS = 25
FRAME_HEIGHT = 512
FRAME_WIDTH = 512
CHUNK_DURATION_SEC = 0.25          # seconds of audio per network send
SAMPLES_PER_CHUNK = int(CHUNK_DURATION_SEC * SAMPLE_RATE)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e8e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Avatar cards */
    .avatar-card {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.08);
        transition: border-color 0.2s, box-shadow 0.2s;
        cursor: pointer;
        background: rgba(255,255,255,0.03);
    }
    .avatar-card:hover {
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.25);
    }
    .avatar-card.selected {
        border-color: #8b5cf6;
        box-shadow: 0 0 24px rgba(139, 92, 246, 0.5);
    }

    /* Gradient heading */
    .gradient-title {
        background: linear-gradient(90deg, #a78bfa 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    /* Status badges */
    .badge-green {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-red {
        background: rgba(239,68,68,0.15);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-yellow {
        background: rgba(234,179,8,0.15);
        color: #eab308;
        border: 1px solid rgba(234,179,8,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.07); }

    /* Section labels */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #a78bfa;
        margin-bottom: 8px;
    }

    /* Generate button */
    .stButton > button {
        background: linear-gradient(90deg, #8b5cf6, #38bdf8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s, transform 0.15s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stButton > button:disabled {
        opacity: 0.4;
        transform: none;
    }

    /* Remove default image caption margin */
    [data-testid="stImage"] > img {
        border-radius: 10px;
    }

    /* Video player border */
    video {
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def http_base_url(ws_uri: str) -> str:
    """Convert ws://host:port → http://host:port."""
    return ws_uri.replace("ws://", "http://").replace("wss://", "https://").rsplit("/ws/", 1)[0]


def fetch_server_avatars(server_uri: str) -> list[str]:
    """Call GET /avatars and return list of avatar names.  Returns [] on failure."""
    try:
        base = http_base_url(server_uri)
        resp = httpx.get(f"{base}/avatars", timeout=3.0)
        resp.raise_for_status()
        return resp.json().get("avatars", [])
    except Exception:
        return []


def local_avatar_images() -> dict[str, str]:
    """
    Scan examples/ directory for image files.
    Returns {stem_name: absolute_path}.
    """
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(EXAMPLES_DIR, pat)))
    paths.sort()
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def local_sample_audios() -> dict[str, str]:
    """Scan examples/ for WAV files. Returns {display_name: path}."""
    paths = glob.glob(os.path.join(EXAMPLES_DIR, "*.wav"))
    paths.sort()
    return {os.path.basename(p): p for p in paths}


def frames_to_mp4_bytes(all_frames: list[np.ndarray], fps: int, audio_path: str) -> bytes:
    """
    Compile a list of (N, H, W, C) uint8 numpy arrays into an MP4 byte blob
    with the original audio stitched in via FFmpeg.
    Returns raw bytes of the final .mp4 file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        silent_mp4 = os.path.join(tmp, "silent.mp4")
        final_mp4 = os.path.join(tmp, "final.mp4")

        # Stack all frame batches
        all_frames_concat = np.concatenate(all_frames, axis=0)  # (T, H, W, C)

        # Write silent video
        with imageio.get_writer(
            silent_mp4,
            format="mp4",
            mode="I",
            fps=fps,
            codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for i in range(all_frames_concat.shape[0]):
                writer.append_data(all_frames_concat[i])

        # Merge audio
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", silent_mp4,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                final_mp4,
            ],
            check=True,
        )

        with open(final_mp4, "rb") as f:
            return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket streaming logic (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

class GenerationResult:
    """Thread-safe container for streaming results."""

    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.error: str | None = None
        self.done: bool = False
        self.frames_received: int = 0
        self._lock = threading.Lock()

    def add_frames(self, frame_batch: np.ndarray):
        with self._lock:
            self.frames.append(frame_batch)
            self.frames_received += frame_batch.shape[0]

    def finish(self):
        with self._lock:
            self.done = True

    def fail(self, msg: str):
        with self._lock:
            self.error = msg
            self.done = True


def _run_async_generation(
    server_uri: str,
    avatar_name: str,
    audio_path: str,
    result: GenerationResult,
):
    """Entry point for the background thread — runs the asyncio event loop."""

    async def _generate():
        try:
            audio_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

            # Pad to exact chunk boundary
            remainder = len(audio_data) % SAMPLES_PER_CHUNK
            if remainder > 0:
                pad_len = SAMPLES_PER_CHUNK - remainder
                audio_data = np.concatenate(
                    [audio_data, np.zeros(pad_len, dtype=audio_data.dtype)]
                )

            chunks = audio_data.reshape(-1, SAMPLES_PER_CHUNK)

            async with websockets.connect(
                server_uri,
                ping_interval=None,
                ping_timeout=None,
                max_size=None,
            ) as ws:
                # Handshake
                await ws.send(json.dumps({"avatar": avatar_name}))

                stop_event = asyncio.Event()

                async def send_audio():
                    try:
                        for chunk in chunks:
                            if stop_event.is_set():
                                break
                            await ws.send(chunk.astype(np.float32).tobytes())
                            # Simulate real-time: slightly faster than real-time
                            await asyncio.sleep(CHUNK_DURATION_SEC * 0.9)
                        await ws.send(b"EOF")
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    except Exception as e:
                        result.fail(f"Send error: {e}")

                async def recv_video():
                    try:
                        while True:
                            raw = await ws.recv()
                            if isinstance(raw, bytes) and len(raw) > 0:
                                batch = np.frombuffer(raw, dtype=np.uint8).reshape(
                                    -1, FRAME_HEIGHT, FRAME_WIDTH, 3
                                )
                                result.add_frames(batch)
                    except websockets.exceptions.ConnectionClosedOK:
                        pass  # Clean server shutdown — expected path
                    except websockets.exceptions.ConnectionClosedError as e:
                        # If we already received frames the server completed its work and
                        # closed the connection (even if the close frame was missing).
                        # Treat this as a successful finish rather than an error.
                        if result.frames_received == 0 and result.error is None:
                            result.fail(f"Connection closed with error: {e}")
                    except Exception as e:
                        result.fail(f"Recv error: {e}")
                    finally:
                        stop_event.set()

                send_task = asyncio.create_task(send_audio())
                recv_task = asyncio.create_task(recv_video())
                await recv_task
                stop_event.set()
                await send_task

        except Exception as e:
            result.fail(str(e))
        finally:
            result.finish()

    asyncio.run(_generate())


# ─────────────────────────────────────────────────────────────────────────────
# UI — Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.5rem; font-weight:700; color:#a78bfa; margin-bottom:0.2rem;'>⚙️ Settings</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    server_uri = st.text_input(
        "Server WebSocket URI",
        value="ws://localhost:8000/ws/stream",
        help="The WebSocket endpoint of the running server.py",
    )

    # Probe server health
    if st.button("🔌 Check Connection"):
        with st.spinner("Checking..."):
            avatars = fetch_server_avatars(server_uri)
        if avatars:
            st.markdown(
                f'<span class="badge-green">🟢 Server online — {len(avatars)} avatar(s) ready</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge-red">🔴 Server offline or not ready</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#6b7280; line-height:1.6;'>"
        "Run <code>python server.py</code> first, then open this app.<br><br>"
        "Each browser tab generates independently — concurrency is automatic."
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI — Main area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<div class='gradient-title'>🎭 SoulX-FlashHead</div>"
    "<div style='color:#9ca3af; font-size:1.05rem; margin-bottom:1.5rem;'>"
    "Real-time talking-head video generation — pick an avatar, upload audio, and generate."
    "</div>",
    unsafe_allow_html=True,
)

# ── Section 1: Avatar Gallery ─────────────────────────────────────────────────
st.markdown("<div class='section-label'>1 · Choose Your Avatar</div>", unsafe_allow_html=True)

local_avatars = local_avatar_images()

if not local_avatars:
    st.warning(f"No avatar images found in `{EXAMPLES_DIR}`. Add `.png` / `.jpg` files there.")
    st.stop()

# Session state: selected avatar
if "selected_avatar" not in st.session_state:
    st.session_state.selected_avatar = list(local_avatars.keys())[0]

# Draw avatar grid (4 columns)
NUM_COLS = min(4, len(local_avatars))
cols = st.columns(NUM_COLS, gap="small")
for idx, (name, img_path) in enumerate(local_avatars.items()):
    col = cols[idx % NUM_COLS]
    with col:
        is_selected = st.session_state.selected_avatar == name
        border_style = "3px solid #8b5cf6" if is_selected else "2px solid rgba(255,255,255,0.1)"
        shadow_style = "0 0 20px rgba(139, 92, 246, 0.5)" if is_selected else "none"

        st.markdown(
            f"<div style='border:{border_style}; border-radius:12px; overflow:hidden; "
            f"box-shadow:{shadow_style}; transition:all 0.2s;'>",
            unsafe_allow_html=True,
        )
        st.image(img_path, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        label = f"✅ {name}" if is_selected else name
        if st.button(label, key=f"avatar_btn_{name}", use_container_width=True):
            st.session_state.selected_avatar = name
            st.rerun()

selected_avatar = st.session_state.selected_avatar
st.markdown(
    f"<div style='color:#a78bfa; font-size:0.9rem; margin-top:4px;'>"
    f"Selected: <strong>{selected_avatar}</strong></div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Section 2: Audio Input ────────────────────────────────────────────────────
st.markdown("<div class='section-label'>2 · Provide Audio</div>", unsafe_allow_html=True)

audio_tab1, audio_tab2 = st.tabs(["📁 Upload WAV File", "🎵 Use Built-in Sample"])

audio_path_to_use: str | None = None
uploaded_tmp_path: str | None = None

with audio_tab1:
    uploaded_audio = st.file_uploader(
        "Upload a WAV file (16 kHz mono recommended)",
        type=["wav"],
        label_visibility="collapsed",
    )
    if uploaded_audio is not None:
        # Save to a named temp file so librosa can load it
        suffix = ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_audio.read())
        tmp.flush()
        tmp.close()
        uploaded_tmp_path = tmp.name
        audio_path_to_use = uploaded_tmp_path
        st.success(f"✅ Loaded: **{uploaded_audio.name}**")

with audio_tab2:
    sample_audios = local_sample_audios()
    if sample_audios:
        selected_sample = st.selectbox(
            "Pick a sample audio",
            options=list(sample_audios.keys()),
            label_visibility="collapsed",
        )
        if audio_path_to_use is None:
            audio_path_to_use = sample_audios[selected_sample]
        st.info(f"Using sample: **{selected_sample}**")
    else:
        st.warning(f"No `.wav` files found in `{EXAMPLES_DIR}`.")

st.markdown("---")

# ── Section 3: Generate ───────────────────────────────────────────────────────
st.markdown("<div class='section-label'>3 · Generate Video</div>", unsafe_allow_html=True)

generate_clicked = st.button(
    "🚀 Generate Talking-Head Video",
    disabled=(audio_path_to_use is None),
    use_container_width=True,
)

# ── Section 4: Progress + Results ─────────────────────────────────────────────
if generate_clicked and audio_path_to_use:
    result = GenerationResult()

    # Kick off background thread
    thread = threading.Thread(
        target=_run_async_generation,
        args=(server_uri, selected_avatar, audio_path_to_use, result),
        daemon=True,
    )
    thread.start()

    # Show live progress
    st.markdown(
        f"<div style='color:#9ca3af; margin-bottom:0.5rem;'>"
        f"Generating for avatar <strong style='color:#a78bfa'>{selected_avatar}</strong> …"
        f"</div>",
        unsafe_allow_html=True,
    )
    status_slot = st.empty()
    progress_bar = st.progress(0, text="Connecting to server…")

    start_ts = time.time()
    chunk_count = 0
    prev_frames = 0

    # Poll until done — Streamlit re-renders on each st.* call inside the loop
    while not result.done:
        time.sleep(0.3)
        frames_now = result.frames_received
        if frames_now != prev_frames:
            chunk_count += 1
            prev_frames = frames_now
        elapsed = time.time() - start_ts

        status_slot.markdown(
            f"<div class='badge-yellow'>⏳ Received {frames_now} frames "
            f"| {elapsed:.1f}s elapsed</div>",
            unsafe_allow_html=True,
        )
        # Indeterminate spin (capped at 95%)
        progress_bar.progress(min(0.95, chunk_count / max(1, chunk_count + 2)), text="Generating…")

    thread.join()
    elapsed_total = time.time() - start_ts

    if result.error:
        progress_bar.empty()
        status_slot.empty()
        st.error(f"❌ Generation failed: {result.error}")
    elif not result.frames:
        progress_bar.empty()
        status_slot.empty()
        st.warning("⚠️ No video frames were received. Check that the server is running and ready.")
    else:
        progress_bar.progress(1.0, text="Done!")
        status_slot.markdown(
            f"<div class='badge-green'>✅ Complete — {result.frames_received} frames "
            f"in {elapsed_total:.1f}s ({result.frames_received / elapsed_total:.1f} fps avg)</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("<div class='section-label'>4 · Your Video</div>", unsafe_allow_html=True)

        with st.spinner("Compiling MP4…"):
            mp4_bytes = frames_to_mp4_bytes(result.frames, TGT_FPS, audio_path_to_use)

        st.video(mp4_bytes)

        st.download_button(
            label="⬇️ Download MP4",
            data=mp4_bytes,
            file_name=f"flashhead_{selected_avatar}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

        # Generation log (collapsible)
        with st.expander("📊 Generation Details", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Frames", result.frames_received)
            col_b.metric("Generation Time", f"{elapsed_total:.1f}s")
            col_c.metric("Avg FPS", f"{result.frames_received / elapsed_total:.1f}")
            st.markdown(f"- **Avatar**: `{selected_avatar}`")
            st.markdown(f"- **Audio source**: `{os.path.basename(audio_path_to_use)}`")
            st.markdown(f"- **Server URI**: `{server_uri}`")

    # Clean up temp file if it was uploaded
    if uploaded_tmp_path and os.path.exists(uploaded_tmp_path):
        try:
            os.remove(uploaded_tmp_path)
        except Exception:
            pass
