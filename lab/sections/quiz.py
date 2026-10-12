"""The Quiz page: a self-grading, randomly-drawn set of conceptual questions."""
import random

import streamlit as st

from ..config import QUIZ_QUESTIONS, QUIZ_LENGTH

def draw_quiz_questions() -> list:
    """Picks QUIZ_LENGTH question ids at random from the bank, in a random order."""
    ids = [q["id"] for q in QUIZ_QUESTIONS]
    random.shuffle(ids)
    return ids[:QUIZ_LENGTH]


def selected_quiz_questions() -> list:
    """The questions for this sitting, in the order they were drawn."""
    by_id = {q["id"]: q for q in QUIZ_QUESTIONS}
    return [by_id[i] for i in st.session_state["quiz_ids"] if i in by_id]


def reset_quiz(new_questions: bool = True):
    """Clears the answers and, by default, draws a fresh set of questions."""
    if new_questions:
        st.session_state["quiz_ids"] = draw_quiz_questions()
    # The radio widgets keep their own state under these keys; drop them so a redraw starts clean.
    for key in [k for k in list(st.session_state) if k.startswith("quiz_radio_")]:
        del st.session_state[key]
    st.session_state["quiz_answers"] = {}
    st.session_state["quiz_submitted"] = False
    st.session_state["quiz_score"] = 0


def render_quiz_section():
    """Renders Section 3: Assessment Quiz with Self-Grading and Feedback."""
    st.header("Quiz", icon=":material/quiz:")

    questions = selected_quiz_questions()

    head_left, head_right = st.columns([3, 1], vertical_alignment="center")
    with head_left:
        if st.session_state.get("quiz_submitted", False):
            score = st.session_state.get("quiz_score", 0)
            with st.container(border=True, horizontal=True, vertical_alignment="center"):
                st.metric("Last score", f"{score} / {QUIZ_LENGTH}")
                st.progress(score / QUIZ_LENGTH, text=f"{score / QUIZ_LENGTH:.0%}")
    with head_right:
        if st.button("New set of questions", icon=":material/casino:", width="stretch"):
            reset_quiz()
            st.rerun()

    with st.form("lab_quiz_form", border=False):
        user_responses = {}
        for number, q in enumerate(questions, start=1):
            with st.container(border=True):
                st.markdown(f":gray[Question {number} of {QUIZ_LENGTH}]  \n**{q['question']}**")
                selected = st.radio(
                    label=f"Options for question {number}",
                    options=q["options"],
                    index=st.session_state["quiz_answers"].get(q["id"], 0),
                    key=f"quiz_radio_{q['id']}",
                    label_visibility="collapsed"
                )
                user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit answers", type="primary", icon=":material/done_all:")

    if submitted:
        score = sum(1 for q in questions if user_responses[q["id"]] == q["answer_index"])
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True
        st.session_state["quiz_score"] = score

        st.subheader("Results", icon=":material/grading:")
        st.progress(score / QUIZ_LENGTH,
                    text=f"**{score} / {QUIZ_LENGTH}** correct ({score / QUIZ_LENGTH:.0%})")
        for number, q in enumerate(questions, start=1):
            user_ans, correct_ans = user_responses[q["id"]], q["answer_index"]
            if user_ans == correct_ans:
                with st.expander(f"Question {number}: correct", icon=":material/check_circle:"):
                    st.markdown(q["explanation"])
            else:
                with st.expander(f"Question {number}: incorrect", icon=":material/cancel:", expanded=True):
                    st.markdown(f":red[Your answer: {q['options'][user_ans]}]  \n"
                                f":green[Correct answer: {q['options'][correct_ans]}]")
                    st.markdown(q["explanation"])
