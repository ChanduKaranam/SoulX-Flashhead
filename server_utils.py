import asyncio
import time
from typing import Dict, Any
import io
import av

class SessionManager:
    def __init__(self):
        # Maps session_id -> dict with user state + av containers
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str, initial_state: Dict[str, Any], fps: int = 25, width: int = 512, height: int = 512, audio_rate: int = 16000):
        """
        initial_state comes from `pipeline.prepare_params_stateless`
        We also initialize an PyAV container for MPEG-TS streaming.
        """
        # Create an in-memory buffer
        memory_buffer = io.BytesIO()
        
        # Open an MPEG-TS container
        container = av.open(memory_buffer, mode='w', format='mpegts')
        
        # Add Video Stream (H.264 is widely supported)
        stream_v = container.add_stream('libx264', rate=fps)
        stream_v.width = width
        stream_v.height = height
        stream_v.pix_fmt = 'yuv420p'
        # Some x264 options for low-latency streaming
        stream_v.options = {
            'preset': 'ultrafast',
            'tune': 'zerolatency',
            'crf': '23'
        }

        # Add Audio Stream (AAC)
        stream_a = container.add_stream('aac', rate=audio_rate)
        
        # Inject the AV stuff into the state payload
        state_with_av = initial_state.copy()
        state_with_av.update({
            "av_buffer": memory_buffer,
            "av_container": container,
            "stream_v": stream_v,
            "stream_a": stream_a,
            "audio_pts": 0,  # track presentation timestamps manually if needed
            "video_pts": 0
        })

        self.sessions[session_id] = state_with_av
        print(f"[SessionManager] Created session {session_id} with PyAV MPEG-TS Muxer.")

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, state: Dict[str, Any]):
        self.sessions[session_id] = state

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            # Safely close the AV container to flush remaining packets
            state = self.sessions[session_id]
            if "av_container" in state:
                try:
                    state["av_container"].close()
                except Exception as e:
                    print(f"Error closing container for {session_id}: {e}")
            
            del self.sessions[session_id]
            print(f"[SessionManager] Deleted session {session_id}")
