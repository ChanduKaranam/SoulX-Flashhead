import asyncio
import websockets
import librosa
import numpy as np
async def mock_client(client_id, audio_path, server_uri, slice_len, sample_rate, tgt_fps):
    """
    Connects to the server and streams audio chunks simulating a real-time microphone.
    """
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    
    # Load raw audio
    print(f"[Client {client_id}] Loading {audio_path}...")
    audio_data, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Pad to ensure even chunks
    remainder = len(audio_data) % human_speech_array_slice_len
    if remainder > 0:
        pad_length = human_speech_array_slice_len - remainder
        audio_data = np.concatenate([audio_data, np.zeros(pad_length, dtype=audio_data.dtype)])

    chunks = audio_data.reshape(-1, human_speech_array_slice_len)

    print(f"[Client {client_id}] Connecting to {server_uri}...")
    # Increase ping timeout so the client doesn't drop the connection while the server GPU is busy
    # Set max_size to None because a chunk of 28 uncompressed RGB frames is ~22MB, far exceeding the 1MB default limit
    
    all_video_frames = []
    
    # We will write the incoming MPEG-TS stream directly to a file
    output_filename = f"client{client_id}_concurrent_test.ts"
    print(f"[Client {client_id}] Connected. Output will be saved to {output_filename}")
    
    # Overwrite the file initially to clear any old runs
    with open(output_filename, "wb") as f:
        pass

    async with websockets.connect(server_uri, ping_interval=70, ping_timeout=70, max_size=None) as websocket:
        print(f"[Client {client_id}] Streaming {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            # 1. Send floating point audio array
            chunk_bytes = chunk.astype(np.float32).tobytes()
            await websocket.send(chunk_bytes)
            
            # 2. Wait for the generated video frames bytes back
            result = await websocket.recv()
            print(f"[Client {client_id}] Received chunk {i+1}/{len(chunks)} ({len(result)} bytes)")
            
            # 3. Append the raw MPEG-TS chunk directly to the file
            if len(result) > 0:
                with open(output_filename, "ab") as f:
                    f.write(result)

            # Simulate real-time streaming delay (the time it takes for a real person to speak the chunk)
            chunk_duration = human_speech_array_slice_len / sample_rate
            await asyncio.sleep(chunk_duration * 0.5) # Speed it up slightly for testing
            
    # When streaming is done, the video is already compiled on disk!
    print(f"[Client {client_id}] SUCCESS! Video fully streamed and saved to {output_filename}")

async def main():
    # Model config constraints (as per infer_params)
    sample_rate = 16000
    tgt_fps = 25
    frame_num = 33
    motion_frames_num = 5
    slice_len = frame_num - motion_frames_num
    
    uri = "ws://localhost:8000/ws/stream"
    audio_file = "examples/podcast_sichuan_16k.wav"

    # Spin up 3 clients simultaneously to prove concurrent execution
    # For safety, we stagger their starts slightly.
    print("Launching 3 concurrent clients...")
    client1 = mock_client(1, audio_file, uri, slice_len, sample_rate, tgt_fps)
    client2 = mock_client(2, audio_file, uri, slice_len, sample_rate, tgt_fps)
    client3 = mock_client(3, audio_file, uri, slice_len, sample_rate, tgt_fps)

    await asyncio.gather(client1, client2, client3)

if __name__ == "__main__":
    asyncio.run(main())
