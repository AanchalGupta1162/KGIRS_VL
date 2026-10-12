"""Session-state initialization, run once per browser session."""
from datetime import datetime

import streamlit as st

from .config import EXPERIMENT_CONFIG, DEFAULT_CORPUS, DEFAULT_QUERY
from .sections.quiz import draw_quiz_questions
from .sections.simulation import DEFAULT_ANIMATION_SPEED

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_ids" not in st.session_state:
        st.session_state["quiz_ids"] = draw_quiz_questions()
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "",
            "id": f"{EXPERIMENT_CONFIG['roll_no']} / Group {EXPERIMENT_CONFIG['group_no']}",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "corpus_text" not in st.session_state:
        st.session_state["corpus_text"] = "\n".join(DEFAULT_CORPUS)
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = DEFAULT_QUERY
    if "anim_step" not in st.session_state:
        st.session_state["anim_step"] = 0
    if "anim_playing" not in st.session_state:
        st.session_state["anim_playing"] = False
    if "anim_fingerprint" not in st.session_state:
        st.session_state["anim_fingerprint"] = None
    if "anim_speed" not in st.session_state:
        st.session_state["anim_speed"] = DEFAULT_ANIMATION_SPEED
