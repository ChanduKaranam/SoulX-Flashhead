import asyncio
import time
from typing import Dict, Any

class SessionManager:
    def __init__(self):
        # Maps session_id -> { "original_color_reference": Tensor, "ref_img_latent": Tensor, "latent_motion_frames": Tensor }
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str, initial_state: Dict[str, Any]):
        """
        initial_state comes from `pipeline.prepare_params_stateless`
        """
        self.sessions[session_id] = initial_state
        print(f"[SessionManager] Created session {session_id}")

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, state: Dict[str, Any]):
        self.sessions[session_id] = state

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            print(f"[SessionManager] Deleted session {session_id}")
