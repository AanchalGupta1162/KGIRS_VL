"""The Report page: student details, observations, checklist, and the PDF download."""
from datetime import datetime

import pandas as pd
import streamlit as st

from ..config import EXPERIMENT_CONFIG, QUIZ_LENGTH, DEFAULT_NOTES
from ..pdf_report import generate_pdf_report

def render_report_section():
    """Renders Section 4: Dynamic Lab Report Generator with Guaranteed PDF Export."""
    st.header("Report", icon=":material/description:")
    st.caption("Fill in your details and observations, check the checklist, then download the PDF.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Student details**")
            c1, c2 = st.columns(2)
            with c1:
                student_name = st.text_input(
                    "Student name", value=st.session_state["student_info"].get("name", ""),
                    placeholder="Your full name"
                )
            with c2:
                student_id = st.text_input(
                    "Roll nos / group",
                    value=st.session_state["student_info"].get(
                        "id", f"{EXPERIMENT_CONFIG['roll_no']} / Group {EXPERIMENT_CONFIG['group_no']}"
                    )
                )
            lab_date = st.date_input("Experiment date", value=datetime.now())

            st.session_state["student_info"]["name"] = student_name
            st.session_state["student_info"]["id"] = student_id
            st.session_state["student_info"]["date"] = str(lab_date)

        with st.container(border=True):
            st.markdown("**Observations and conclusions**")
            student_notes = st.text_area(
                "Observations", label_visibility="collapsed",
                value=st.session_state.get("student_notes") or DEFAULT_NOTES, height=160
            )
            st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()
    quiz_score = st.session_state.get("quiz_score", 0)

    with right:
        with st.container(border=True):
            st.markdown("**Report checklist**")
            n_trials = len(trials_df)
            st.markdown(
                f"{':material/check_circle:' if student_name.strip() else ':material/radio_button_unchecked:'} Student name  \n"
                f"{':material/check_circle:' if n_trials >= 3 else ':material/radio_button_unchecked:'} "
                f"Trials recorded :gray[({n_trials}, 3+ recommended)]  \n"
                f"{':material/check_circle:' if st.session_state.get('quiz_submitted') else ':material/radio_button_unchecked:'} "
                f"Quiz submitted :gray[({quiz_score} / {QUIZ_LENGTH})]"
            )

        pdf_bytes = generate_pdf_report(
            student_name=student_name,
            student_id=student_id,
            date_str=str(lab_date),
            trials_df=trials_df,
            quiz_score=quiz_score,
            quiz_total=QUIZ_LENGTH,
            student_notes=student_notes
        )

        # The PDF is only sent to this viewer's browser; it is never written to disk, where every
        # user of a hosted app would share (and could open) the same file.
        st.download_button(
            label="Download lab report (PDF)", data=pdf_bytes, file_name="lab_report.pdf",
            mime="application/pdf", key="stream_pdf_btn", type="primary",
            icon=":material/download:", width="stretch"
        )

    st.subheader("Recorded trials", icon=":material/table_rows:")
    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, width="stretch")
    else:
        st.caption("No trials recorded yet. The report will list 0 trials. Record some on the Simulation page.")
