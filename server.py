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

        state = session_manager.get_session(session_id)
        if not state:
            generation_queue.task_done()
            continue

        try:
            # Prepare streaming audio indices exactly as in the original generate_video script stream mode
            audio_end_idx = cached_audio_duration * tgt_fps
            audio_start_idx = audio_end_idx - frame_num
            
            # Extract audio embeddings
            # (Note: audio_embedding uses the pipeline's internal Wav2Vec extractor)
            audio_embedding = get_audio_embedding(pipeline, audio_chunk, audio_start_idx, audio_end_idx)

            # Generate video chunk statelessly
            video_chunk, updated_state = pipeline.generate_stateless(audio_embedding, state)
            
            # Only return the non-motion frames to the user (the actual new generated sequence)
            video_chunk = video_chunk[motion_frames_num:]
            
            # Save updated state
            session_manager.update_session(session_id, updated_state)

            # Serialize and send back the frame bytes
            # For this MVP, we convert the torch tensor to a raw byte array
            # In production, this would be encoded to h264 chunks via ffmpeg or PyAV
            frames_np = video_chunk.cpu().numpy().astype(np.uint8)
            await ws.send_bytes(frames_np.tobytes())

        except Exception as e:
            logger.error(f"Error in generation worker: {str(e)}")
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
    
    # Start the continuous worker
    asyncio.create_task(generation_worker())

@app.websocket("/ws/stream")
async def stream_websocket(ws: WebSocket):
    await ws.accept()
    session_id = str(id(ws)) # Simple unique session ID

    # Read the initial payload containing the condition image
    # Note: For true production, you would send the image bytes over WS. 
    # Here we simulate by reading the examples/girl.png path just to test the concurrent engine.
    initial_state_dict = pipeline.prepare_params_stateless(
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
    
    # Extract the state for the 'girl' image specifically
    state = initial_state_dict["girl"]
    session_manager.create_session(session_id, state)

    # Deque to handle sliding window audio padding, exact same as `generate_video.py`
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

    try:
        while True:
            # 1. Receive the audio segment from the client.
            # Expecting raw float32 bytes for the audio array.
            data = await ws.receive_bytes()
            chunk_array = np.frombuffer(data, dtype=np.float32)

            # Pad if the client sent a chunk slightly smaller than expected
            remainder = len(chunk_array) % human_speech_array_slice_len
            if remainder > 0:
                pad_length = human_speech_array_slice_len - remainder
                chunk_array = np.concatenate([chunk_array, np.zeros(pad_length, dtype=chunk_array.dtype)])
            
            # 2. Append to rolling deque
            audio_dq.extend(chunk_array.tolist())
            audio_array = np.array(audio_dq)

            # 3. Queue the generation request
            # This is non-blocking right here; we just wait for the worker to pick it up.
            await generation_queue.put({
                "session_id": session_id,
                "audio_chunk": audio_array,
                "websocket": ws
            })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
        session_manager.delete_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket Error: {str(e)}")
        session_manager.delete_session(session_id)

if __name__ == "__main__":
    import uvicorn
    # explicitly force the 'websockets' library for handling WS connections since Vast.ai drops it
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="websockets")
