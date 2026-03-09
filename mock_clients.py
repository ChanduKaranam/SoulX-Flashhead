import asyncio
import websockets
import librosa
import numpy as np
import imageio
import subprocess
import os

async def mock_client(client_id, avatar_name, audio_path, server_uri, chunk_duration_sec, sample_rate, tgt_fps):
    """
    Connects to the server, sends an avatar configuration, and streams generic audio chunks simulating a real-time microphone.
    """
    # Simply stream raw audio in N-second increments. 
    # The server is now responsible for buffering and slicing it to exactly what the model needs.
    samples_per_chunk = int(chunk_duration_sec * sample_rate)
    
    # Load raw audio
    print(f"[Client {client_id}] Loading avatar '{avatar_name}' with audio '{audio_path}'...")
    audio_data, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Pad to ensure even chunks
    remainder = len(audio_data) % samples_per_chunk
    if remainder > 0:
        pad_length = samples_per_chunk - remainder
        audio_data = np.concatenate([audio_data, np.zeros(pad_length, dtype=audio_data.dtype)])

    chunks = audio_data.reshape(-1, samples_per_chunk)

    print(f"[Client {client_id}] Connecting to {server_uri}...")
    # Increase ping timeout so the client doesn't drop the connection while the server GPU is busy
    # Set max_size to None because a chunk of 28 uncompressed RGB frames is ~22MB, far exceeding the 1MB default limit
    
    all_video_frames = []
    # We DISABLE ping_interval (None) because Uvicorn's websockets auto-pong task crashes
    # when it collides with a massive video payload chunk send_bytes call.
    async with websockets.connect(server_uri, ping_interval=None, ping_timeout=None, max_size=None) as websocket:
        import json
        
        # Handshake: Send avatar configuration first
        config = {"avatar": avatar_name}
        await websocket.send(json.dumps(config))
        
        print(f"[Client {client_id}] Connected. Streaming {len(chunks)} chunks...")

        stop_event = asyncio.Event()

        async def send_audio():
            try:
                for i, chunk in enumerate(chunks):
                    if stop_event.is_set():
                        break
                    # 1. Send floating point audio array
                    chunk_bytes = chunk.astype(np.float32).tobytes()
                    await websocket.send(chunk_bytes)
                    
                    # Simulate real-time streaming delay (the time it takes for a real person to speak the chunk)
                    await asyncio.sleep(chunk_duration_sec * 0.9) # Slightly faster than real-time

                print(f"[Client {client_id}] Finished streaming audio. Awaiting final video frames...")
                await websocket.send(b"EOF")
            except websockets.exceptions.ConnectionClosed:
                print(f"[Client {client_id}] Server closed connection during send.")
            except Exception as e:
                print(f"[Client {client_id}] Send Error: {e}")

        async def recv_video():
            try:
                while True:
                    # 2. Wait for the generated video frames bytes back
                    result = await websocket.recv()
                    print(f"[Client {client_id}] Received video chunk ({len(result)} bytes)")
                    
                    # 3. Append the raw frames for local compilation
                    if len(result) > 0:
                        frame_array = np.frombuffer(result, dtype=np.uint8)
                        frame_array = frame_array.reshape(-1, 512, 512, 3)
                        all_video_frames.append(frame_array)
            except websockets.exceptions.ConnectionClosedOK:
                print(f"[Client {client_id}] Server successfully closed connection. Generation complete.")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"[Client {client_id}] Server closed connection with error: {e}")
            except Exception as e:
                print(f"[Client {client_id}] Recv Error: {e}")
            finally:
                stop_event.set()

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(recv_video())

        # Wait until the receiver terminates (when the server closes the socket after flushing)
        await recv_task
        # Clean up sender if it was somehow still alive
        stop_event.set()
        await send_task
            
    # When streaming is done, compile the video
    if all_video_frames:
        temp_video = f"client{client_id}_temp.mp4"
        final_video = f"client{client_id}_concurrent_test.mp4"
        
        print(f"[Client {client_id}] Compiling saved chunks into {temp_video} ...")
        final_video_sequence = np.concatenate(all_video_frames, axis=0)
        imageio.mimwrite(temp_video, final_video_sequence, fps=tgt_fps, quality=8)
        
        print(f"[Client {client_id}] Stitching original audio to {final_video} via FFmpeg ...")
        # Use FFmpeg to combine the temp video completely seamlessly with the source audio
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", temp_video,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            final_video
        ], check=True)
        
        # Clean up the silent temp video
        if os.path.exists(temp_video):
            os.remove(temp_video)
            
        print(f"[Client {client_id}] SUCCESS! Video saved to {final_video}")

async def main():
    # Only need raw audio properties here. 
    # frame_num, motion_frames_num, etc are entirely backend concerns now.
    sample_rate = 16000
    tgt_fps = 25
    chunk_duration_sec = 0.25 # Send a quarter second of audio every network ping
    
    uri = "ws://localhost:8000/ws/stream"
    
    # Mock heterogeneous configuration
    # Note: These image basenames MUST exist in the server's 'examples/' directory
    # Note: These audio files must exist locally
    configs = [
        {"id": 1, "avatar": "girl", "audio": "examples/podcast_sichuan_16k.wav"},
        {"id": 2, "avatar": "boy", "audio": "examples/podcast_sichuan_16k.wav"},    # Fallback to same audio if you don't have multiple
        {"id": 3, "avatar": "woman", "audio": "examples/podcast_sichuan_16k.wav"}, 
    ]

    # Spin up 3 clients simultaneously to prove concurrent execution
    # For safety, we stagger their starts slightly.
    print("Launching concurrent clients with heterogeneous configurations...")
    clients = [
        mock_client(conf["id"], conf["avatar"], conf["audio"], uri, chunk_duration_sec, sample_rate, tgt_fps)
        for conf in configs
    ]

    await asyncio.gather(*clients)

if __name__ == "__main__":
    asyncio.run(main())
