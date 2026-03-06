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

# Global generation queue to serialize GPU requests for the single pipeline
generation_queue = asyncio.Queue()

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
    Background worker that continuously pulls chunk requests from the queue and 
    processes them one by one blazing fast using the stateless pipeline.
    """
    logger.info("Generation Worker Started.")
    while True:
        request = await generation_queue.get()
        session_id = request["session_id"]
        audio_chunk = request["audio_chunk"]  # np.array
        ws = request["websocket"]

        from starlette.websockets import WebSocketState
        
        state = session_manager.get_session(session_id)
        # Check if session is deleted, OR if the websocket has disconnected
        if not state or ws.client_state == WebSocketState.DISCONNECTED:
            generation_queue.task_done()
            continue

        try:
            # Prepare streaming audio indices exactly as in the original generate_video script stream mode
            audio_end_idx = cached_audio_duration * tgt_fps
            audio_start_idx = audio_end_idx - frame_num
            
            # Extract audio embeddings
            # (Note: audio_embedding uses the pipeline's internal Wav2Vec extractor)
            audio_embedding = get_audio_embedding(pipeline, audio_chunk, audio_start_idx, audio_end_idx)
            
            logger.info(f"[Audio Sync Debug] audio_chunk std: {audio_chunk.std():.5f}, max: {audio_chunk.max():.5f}")
            logger.info(f"[Audio Sync Debug] audio_embedding shape: {audio_embedding.shape}, std: {audio_embedding.std().item():.5f}")

            # Generate video chunk statelessly - MUST BE RUN IN A THREAD!
            # If we run PyTorch directly here, it blocks the entire FastAPI asyncio loop
            # and prevents WebSockets from successfully sending/receiving keepalive pings.
            def _sync_generate():
                return pipeline.generate_stateless(audio_embedding, state)
            
            video_chunk, updated_state = await asyncio.to_thread(_sync_generate)
            # 1. ENCODE VIDEO FRAMES
            # video_chunk shape is (Channels, Time, Height, Width) -> we need (Time, Height, Width, Channels) for av
            # We slice the Time dimension to remove the prepended context frames
            video_chunk = video_chunk[:, motion_frames_num:, :, :]
            
            # Normalize the PyTorch tensor from [-1, 1] float to [0, 255] uint8 RGB
            # Math adapted directly from inference.run_pipeline
            # shape was (C, T, H, W) -> we need (T, H, W, C) for imageio/cv2
            normalized_frames = (((video_chunk + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255).contiguous()

            # Serialize and send back the frame bytes
            frames_np = normalized_frames.cpu().numpy().astype(np.uint8)
            frame_bytes = frames_np.tobytes()
            
            # Place the finalized frames back into the user's dedicated response queue
            response_queue = state.get("response_queue")
            if response_queue:
                response_queue.put_nowait(frame_bytes)

        except Exception as e:
            import traceback
            logger.error(f"Error in generation worker: {e}\n{traceback.format_exc()}")
        finally:
            generation_queue.task_done()

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
    logger.info("Pre-computing initial latents for 'examples/girl.png'...")
    global global_initial_state_dict
    global_initial_state_dict = pipeline.prepare_params_stateless(
        cond_image_path_or_dir="examples/girl.png",
        target_size=(infer_params['height'], infer_params['width']),
        frame_num=infer_params['frame_num'],
        motion_frames_num=infer_params['motion_frames_num'],
        sampling_steps=infer_params['sample_steps'],
        seed=42,
        shift=infer_params['sample_shift'],
        color_correction_strength=infer_params['color_correction_strength'],
        use_face_crop=False
    )
    logger.info("Initial latent state cached. Ready for client connections.")

    # Start the continuous worker
    asyncio.create_task(generation_worker())

@app.websocket("/ws/stream")
async def stream_websocket(ws: WebSocket):
    await ws.accept()
    session_id = str(id(ws)) # Simple unique session ID

    # Use the globally pre-compiled state so connection is instant
    state = global_initial_state_dict["girl"]
    
    # We must deep copy the state because multiple users cannot modify the same latents dictionary in memory
    import copy
    user_state = copy.deepcopy(state)
    
    # Create the decoupled queue for this exact user session
    response_queue = asyncio.Queue()
    user_state["response_queue"] = response_queue
    
    session_manager.create_session(session_id, user_state)

    # Deque to handle sliding window audio padding
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

    async def receive_audio():
        """Listen strictly for incoming audio chunks from the client."""
        try:
            while True:
                data = await ws.receive_bytes()
                chunk_array = np.frombuffer(data, dtype=np.float32)

                # Pad if the client sent a chunk slightly smaller than expected
                remainder = len(chunk_array) % human_speech_array_slice_len
                if remainder > 0:
                    pad_length = human_speech_array_slice_len - remainder
                    chunk_array = np.concatenate([chunk_array, np.zeros(pad_length, dtype=chunk_array.dtype)])
                
                # Append to rolling deque
                audio_dq.extend(chunk_array.tolist())
                audio_array = np.array(audio_dq)

                # Toss the request onto the global generation fire
                await generation_queue.put({
                    "session_id": session_id,
                    "audio_chunk": audio_array,
                    "websocket": ws # worker doesn't use this anymore, but leaving for legacy/state tracking
                })
        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected normally during receive.")
        except Exception as e:
            logger.error(f"Error in receive loop for {session_id}: {e}")

    async def send_video():
        """Listen strictly for generated video frames and send them back."""
        try:
            while True:
                # Wait for the worker to grant us our generated chunk bytes
                video_bytes = await response_queue.get()
                await ws.send_bytes(video_bytes)
                response_queue.task_done()
        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected normally during send.")
        except RuntimeError:
             # Client closed mid-dispatch
             pass
        except Exception as e:
            logger.error(f"Error in send loop for {session_id}: {e}")

    # Run both the Receiver and the Sender concurrently linked to this active route scope!
    # By using FIRST_COMPLETED, if the client drops connection and receive_audio() throws 
    # a WebSocketDisconnect, the gather finishes and cleans up both tasks instantly.
    try:
        receive_task = asyncio.create_task(receive_audio())
        send_task = asyncio.create_task(send_video())
        
        await asyncio.wait([receive_task, send_task], return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel whatever didn't finish
        receive_task.cancel()
        send_task.cancel()

    finally:
        # Tear down! We only ever clean up here, when the ASGI route definitively dies.
        logger.info(f"Tearing down session: {session_id}")
        session_manager.delete_session(session_id)

if __name__ == "__main__":
    import uvicorn
    # explicitly force the 'websockets' library for handling WS connections since Vast.ai drops it
    # We also extend the ping timeouts massively because the GPU batching might stall the main event loop
    # We remove the max_size limit (None) because raw 28-frame chunks are ~22MB, shattering the 1MB default
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="websockets", ws_ping_interval=70, ws_ping_timeout=70, ws_max_size=None)
