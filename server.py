import os
import asyncio
import numpy as np
import argparse
from loguru import logger
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from collections import deque

from flash_head.inference import get_pipeline, get_infer_params, get_audio_embedding
from server_utils import SessionManager

app = FastAPI()

# Global state hold
pipeline = None
session_manager = SessionManager()
global_initial_state_dict = None

# Per-session generation queues for round-robin scheduling.
# Each active WebSocket session gets its own queue so no single user
# monopolises the GPU worker — every session gets a turn each cycle.
session_queues: dict = {}  # session_id -> asyncio.Queue

# Inference parameters to be initialized after pipeline loads
infer_params = None
sample_rate = 16000
tgt_fps = 25
cached_audio_duration = 0
frame_num = 0
motion_frames_num = 0
slice_len = 0
human_speech_array_slice_len = 0

async def generation_worker():
    """
    Round-robin generation worker.

    Instead of a single global FIFO queue (which causes one user to monopolise
    the GPU while others wait), each session registers its own queue.  The
    worker cycles through every active session on each iteration, processing
    exactly ONE chunk per session before moving to the next.  This interleaves
    GPU work evenly so all concurrent users see frames arriving at the same time.
    """
    from starlette.websockets import WebSocketState

    logger.info("Round-Robin Generation Worker Started.")
    while True:
        # Snapshot of active session IDs for this cycle
        active_ids = list(session_queues.keys())

        if not active_ids:
            # Nothing to do — yield control to the event loop briefly
            await asyncio.sleep(0.01)
            continue

        any_processed = False

        for session_id in active_ids:
            q = session_queues.get(session_id)
            if q is None or q.empty():
                continue

            try:
                request = q.get_nowait()
            except asyncio.QueueEmpty:
                continue

            any_processed = True

            # ── EOF marker: signal the response queue to close ──────────────
            if request.get("EOF"):
                state = session_manager.get_session(session_id)
                if state and state.get("response_queue"):
                    state["response_queue"].put_nowait(b"EOF")
                q.task_done()
                continue

            # ── Normal audio chunk ──────────────────────────────────────────
            audio_chunk = request["audio_chunk"]
            ws = request["websocket"]

            state = session_manager.get_session(session_id)
            if not state or ws.client_state == WebSocketState.DISCONNECTED:
                q.task_done()
                continue

            try:
                audio_end_idx = cached_audio_duration * tgt_fps
                audio_start_idx = audio_end_idx - frame_num

                audio_embedding = get_audio_embedding(
                    pipeline, audio_chunk, audio_start_idx, audio_end_idx
                )

                logger.info(
                    f"[{session_id}] audio_chunk std: {audio_chunk.std():.5f}  "
                    f"embedding shape: {audio_embedding.shape}"
                )

                def _sync_generate():
                    return pipeline.generate_stateless(audio_embedding, state)

                video_chunk, updated_state = await asyncio.to_thread(_sync_generate)

                video_chunk = video_chunk[:, motion_frames_num:, :, :]
                normalized_frames = (
                    ((video_chunk + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255
                ).contiguous()

                frames_np = normalized_frames.cpu().numpy().astype(np.uint8)
                frame_bytes = frames_np.tobytes()

                response_queue = state.get("response_queue")
                if response_queue:
                    response_queue.put_nowait(frame_bytes)

            except Exception as e:
                import traceback
                logger.error(
                    f"Error processing chunk for {session_id}: {e}\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                q.task_done()

        if not any_processed:
            # All queues were empty this cycle — yield briefly
            await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup_event():
    global pipeline, infer_params, sample_rate, tgt_fps, cached_audio_duration
    global frame_num, motion_frames_num, slice_len, human_speech_array_slice_len

    logger.info("Loading Pipeline into GPU...")
    # These paths are based on the original README setup
    ckpt_dir = "models/SoulX-FlashHead-1_3B"
    wav2vec_dir = "models/wav2vec2-base-960h"
    model_type = "lite" 

    pipeline = get_pipeline(world_size=1, ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir, model_type=model_type)
    
    # Now that pipeline is initialized, inference.py has computed the internal params
    infer_params = get_infer_params()
    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps

    logger.info(f"Pipeline Loaded Successfully. Spinning up worker. Slice size: {slice_len}")
    
    # Pre-compute the initial state dictionary globally so it doesn't block WebSockets
    logger.info("Scanning 'examples/' directory and pre-computing initial latents for all avatars...")
    global global_initial_state_dict
    
    # We pass the directory path instead of a single file path.
    # get_cond_image_dict inside pipeline.prepare_params_stateless will automatically
    # find all PNGs and return a dictionary mapping `filename -> config`.
    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        logger.warning(f"Warning: Directory '{examples_dir}' not found. Initializing empty state.")
        global_initial_state_dict = {}
    else:
        global_initial_state_dict = pipeline.prepare_params_stateless(
            cond_image_path_or_dir=examples_dir,
            target_size=(infer_params['height'], infer_params['width']),
            frame_num=infer_params['frame_num'],
            motion_frames_num=infer_params['motion_frames_num'],
            sampling_steps=infer_params['sample_steps'],
            seed=42,
            shift=infer_params['sample_shift'],
            color_correction_strength=infer_params['color_correction_strength'],
            use_face_crop=False
        )
        
    loaded_avatars = list(global_initial_state_dict.keys())
    logger.info(f"Initial latent state cached for avatars: {loaded_avatars}. Ready for client connections.")

    # Start the continuous worker
    asyncio.create_task(generation_worker())

@app.get("/avatars")
async def list_avatars():
    """Returns the list of avatar names that have been pre-computed and are ready for streaming."""
    return {"avatars": list(global_initial_state_dict.keys())}


@app.websocket("/ws/stream")
async def stream_websocket(ws: WebSocket):
    await ws.accept()
    session_id = str(id(ws)) # Simple unique session ID

    # Read the first message as a configuration object to determine which avatar to use
    try:
        import json
        config_msg = await ws.receive_text()
        config = json.loads(config_msg)
        avatar_name = config.get("avatar", "girl") # Default to girl if not specified
    except Exception as e:
        logger.error(f"Client {session_id} failed handshake parsing: {e}")
        # Removed `reason` kwarg as older starlette versions may throw TypeError
        await ws.close(code=1003)
        return

    logger.info(f"Client {session_id} connected. Requested avatar: '{avatar_name}'")

    if not global_initial_state_dict:
        logger.error("Server has NO pre-computed avatars available!")
        await ws.close(code=1011)
        return

    if avatar_name not in global_initial_state_dict:
        logger.warning(f"Requested avatar '{avatar_name}' not found in cached initial states! Available: {list(global_initial_state_dict.keys())}")
        fallback_avatar = list(global_initial_state_dict.keys())[0]
        logger.warning(f"Falling back gracefully to: '{fallback_avatar}'")
        avatar_name = fallback_avatar

    # Use the globally pre-compiled state so connection is instant
    state = global_initial_state_dict[avatar_name]
    
    # We must clone the state because multiple users cannot modify the same latents tensor in memory!
    # Explicitly using .clone() instead of copy.deepcopy to prevent obscure Starlette route crashes
    user_state = {
        "original_color_reference": state["original_color_reference"].clone(),
        "ref_img_latent": state["ref_img_latent"].clone(),
        "latent_motion_frames": state["latent_motion_frames"].clone()
    }
    
    # OVERRIDE the copied generator state with a brand new seed just for this user!
    import random
    user_seed = random.randint(0, 1000000)
    user_gen = torch.Generator(device=pipeline.device).manual_seed(user_seed)
    user_state["generator_state"] = user_gen.get_state().cpu()
    
    # Create the decoupled response queue and per-session generation queue
    response_queue = asyncio.Queue()
    user_state["response_queue"] = response_queue

    session_manager.create_session(session_id, user_state)

    # Register this session in the round-robin scheduler
    session_queues[session_id] = asyncio.Queue()

    # Deque to handle sliding window audio padding (exact same as original generate_video.py)
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
    
    # Decouple incoming network bytes from the strict pipeline slice length
    session_audio_buffer = []

    stop_event = asyncio.Event()

    async def receive_audio():
        """Listen strictly for incoming audio chunks from the client."""
        try:
            while not stop_event.is_set():
                data = await ws.receive_bytes()

                if data == b"EOF":
                    logger.info(f"Client {session_id} explicitly signaled EOF. Flushing buffer.")
                    if len(session_audio_buffer) > 0:
                        remainder = len(session_audio_buffer) % human_speech_array_slice_len
                        if remainder > 0:
                            pad_length = human_speech_array_slice_len - remainder
                            session_audio_buffer.extend([0.0] * pad_length)
                        exact_slice = session_audio_buffer[:human_speech_array_slice_len]
                        audio_dq.extend(exact_slice)
                        audio_array = np.array(audio_dq)
                        q = session_queues.get(session_id)
                        if q is not None:
                            await q.put({
                                "session_id": session_id,
                                "audio_chunk": audio_array,
                                "websocket": ws
                            })
                    # Pilot EOF to per-session queue
                    q = session_queues.get(session_id)
                    if q is not None:
                        await q.put({"session_id": session_id, "EOF": True})
                    continue  # Hang here until the server ultimately closes the socket for us!

                # 1. Append raw bytes to session buffer
                chunk_array = np.frombuffer(data, dtype=np.float32)
                session_audio_buffer.extend(chunk_array.tolist())

                # 2. Consume exactly `human_speech_array_slice_len` from the buffer when available
                while len(session_audio_buffer) >= human_speech_array_slice_len:
                    # Slice exactly the length we need
                    exact_slice = session_audio_buffer[:human_speech_array_slice_len]

                    # Remove the consumed part from the buffer
                    del session_audio_buffer[:human_speech_array_slice_len]

                    # Push it into the model's sliding window
                    audio_dq.extend(exact_slice)
                    audio_array = np.array(audio_dq)

                    # Enqueue to this session's own round-robin queue
                    q = session_queues.get(session_id)
                    if q is not None:
                        await q.put({
                            "session_id": session_id,
                            "audio_chunk": audio_array,
                            "websocket": ws
                        })
        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected normally during receive. They did not wait for the end.")
        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"Error in receive loop for {session_id}: {e}")
        finally:
            stop_event.set()

    async def send_video():
        """Listen strictly for generated video frames and send them back."""
        try:
            while not stop_event.is_set():
                # Wait with a timeout so we can periodically check stop_event
                try:
                    video_bytes = await asyncio.wait_for(response_queue.get(), timeout=1.0)
                    if video_bytes == b"EOF":
                        logger.info(f"All video generated for {session_id}. Closing output loop gracefully.")
                        break
                    await ws.send_bytes(video_bytes)
                    response_queue.task_done()
                except asyncio.TimeoutError:
                    continue
        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected normally during send.")
        except RuntimeError:
             pass # Client closed mid-dispatch
        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"Error in send loop for {session_id}: {e}")
        finally:
            stop_event.set()

    receive_task = None
    send_task = None
    try:
        receive_task = asyncio.create_task(receive_audio())
        send_task = asyncio.create_task(send_video())

        await asyncio.wait([receive_task, send_task], return_when=asyncio.FIRST_COMPLETED)

        # Signal both coroutines to exit cleanly
        stop_event.set()

        # Cancel whichever task is still pending so it doesn't linger as an
        # orphan holding a reference to a socket that Starlette is about to
        # close — this was the root cause of the 'no close frame' error.
        for task in (receive_task, send_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(receive_task, send_task, return_exceptions=True)

    finally:
        # Remove session from round-robin scheduler BEFORE closing the socket
        # so the worker won't try to process stale entries for this session.
        session_queues.pop(session_id, None)
        session_manager.delete_session(session_id)
        logger.info(f"Tearing down session: {session_id}")

        # Send a proper WebSocket close frame so the client receives
        # ConnectionClosedOK instead of the abrupt 'no close frame' error.
        try:
            await ws.close(code=1000)
        except Exception:
            pass  # Already closed, nothing to do

if __name__ == "__main__":
    import uvicorn
    # explicitly force the 'websockets' library for handling WS connections since Vast.ai drops it
    # We DISABLE ping_interval (None) because Uvicorn's background healthcheck task has a known concurrency
    # bug where it tries to write a PING/PONG frame to the raw socket at the exact same moment that our
    # foreground send_video() task is writing a 22MB uncompressed video payload, causing an AssertionError.
    # We remove the max_size limit (None) because raw 28-frame chunks are ~22MB, shattering the 1MB default
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="websockets", ws_ping_interval=None, ws_ping_timeout=None, ws_max_size=None)
