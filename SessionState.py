import streamlit.report_thread as ReportThread
from streamlit.server.server import Server
import streamlit as st

import streamlit as st

def get_session_state(**kwargs):
    """Gets the session state for the current session, creating it if needed."""
    session_id = st.report_thread.get_report_ctx().session_id
    if not hasattr(get_session_state, "states"):
        get_session_state.states = {}
    if session_id not in get_session_state.states:
        get_session_state.states[session_id] = SessionState(**kwargs)
    return get_session_state.states[session_id]

class SessionState(object):
    """Stores the state for a Streamlit session."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
