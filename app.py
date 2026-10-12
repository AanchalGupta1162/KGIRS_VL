"""
Virtual Laboratory - Experiment 4: TF-IDF Based Document Retrieval
Course: Knowledge Graphs and Information Retrieval Systems
Roll Nos: 16-20 | Group No: 4

Built on the standard 4-section Virtual Lab template:
  1. Theory: Concepts, objectives, procedure, and terminology.
  2. Simulation: Interactive TF-IDF retrieval sandbox with a live corpus/query engine.
  3. Quiz: Self-grading conceptual assessment with instant feedback.
  4. Report Generation: Student info, recorded trials, observations, and downloadable PDF report.

Note: No custom CSS is used so that Streamlit's native light and dark themes render seamlessly.

This file is only the entrypoint and page registry. Each page's logic lives in lab/sections/,
shared content and config in lab/config.py, the retrieval engine in lab/engine.py, drawings in
lab/graphics.py, and the PDF exporter in lab/pdf_report.py - see lab/ for the implementation.
"""
import streamlit as st

from lab.state import init_session_state
from lab.sidebar import SIDEBAR_ORDER_CSS, render_sidebar_progress
from lab.sections.aim import render_aim_section
from lab.sections.theory import render_theory_section
from lab.sections.references import render_references_section
from lab.sections.simulation import render_simulation_section
from lab.sections.quiz import render_quiz_section
from lab.sections.report import render_report_section


def main():
    st.set_page_config(
        page_title="TF-IDF Document Retrieval | KGIRS Virtual Lab",
        page_icon=":material/manage_search:",
        layout="wide"
    )

    init_session_state()
    st.markdown(SIDEBAR_ORDER_CSS, unsafe_allow_html=True)

    page = st.navigation([
        st.Page(render_aim_section, title="Purpose", icon=":material/flag:", url_path="aim", default=True),
        st.Page(render_theory_section, title="Theory", icon=":material/menu_book:", url_path="theory"),
        st.Page(render_references_section, title="References", icon=":material/library_books:",
                url_path="references"),
        st.Page(render_simulation_section, title="Simulation", icon=":material/manage_search:", url_path="simulation"),
        st.Page(render_quiz_section, title="Quiz", icon=":material/quiz:", url_path="quiz"),
        st.Page(render_report_section, title="Report", icon=":material/description:", url_path="report"),
    ], position="sidebar")
    st.markdown("### Experiment 4 : TF-IDF Based Document Retrieval")
    page.run()
    render_sidebar_progress()  # after the page so counts include this run's actions


if __name__ == "__main__":
    main()
