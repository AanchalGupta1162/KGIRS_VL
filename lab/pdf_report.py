"""Compiles a session's recorded trials, quiz score and notes into a downloadable PDF."""
import math
from datetime import datetime

import numpy as np
import pandas as pd
from fpdf import FPDF

from .config import EXPERIMENT_CONFIG

PDF_CHAR_MAP = str.maketrans({"—": "-", "–": "-", "‘": "'", "’": "'",
                              "“": '"', "”": '"', "…": "...", "•": "-", " ": " "})


class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                         trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                         student_notes: str) -> bytes:
    """Compiles experiment trial records into a formatted PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, EXPERIMENT_CONFIG["course"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, pdf.get_y(), 190, 22, "FD")
    box_top = pdf.get_y()

    pdf.set_xy(14, box_top + 2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Roll Nos / Group:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", 1)

    pdf.set_xy(14, box_top + 10)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_total and quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({int((quiz_score / quiz_total) * 100 if quiz_total else 0)}%)", 1)

    pdf.set_xy(10, box_top + 24)
    pdf.ln(4)

    # 1. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Recorded Experimental Trials & Data", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No simulation trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        from fpdf.fonts import FontFace

        cols = list(trials_df.columns)
        # Relative widths: long text columns (query, schemes) get more room; cells wrap instead of truncating.
        width_weights = {"Trial #": 1.1, "Query": 3.8, "TF Scheme": 3.0, "IDF Scheme": 3.6,
                         "Cosine Norm.": 1.4, "Top Doc": 1.2, "Top Score": 1.5, "k": 0.7,
                         "P@k": 1.4, "R@k": 1.4, "F1@k": 1.4, "AP": 1.4, "nDCG@k": 1.5, "Timestamp": 1.7}
        col_widths = [width_weights.get(c, 2.0) for c in cols]

        def pdf_text(val) -> str:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                text = "-"
            elif isinstance(val, (bool, np.bool_)):
                text = "Yes" if val else "No"
            elif isinstance(val, float):
                text = f"{val:.4f}"
            else:
                text = str(val)
            # Built-in Helvetica only covers Latin-1: map common typographic characters, replace the rest.
            text = text.translate(PDF_CHAR_MAP)
            return text.encode("latin-1", "replace").decode("latin-1")

        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        with pdf.table(
            col_widths=col_widths,
            text_align="CENTER",
            line_height=4.2,
            padding=1.2,
            borders_layout="ALL",
            cell_fill_color=(248, 250, 252),
            cell_fill_mode="ROWS",
            headings_style=FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(37, 99, 235)),
            repeat_headings=1,
        ) as table:
            header = table.row()
            for c in cols:
                header.cell(pdf_text(c))
            for _, row in trials_df.iterrows():
                data_row = table.row()
                for c in cols:
                    data_row.cell(pdf_text(row[c]))
    pdf.ln(5)

    # 2. Discussion & Notes
    if pdf.get_y() > pdf.h - 45:  # keep the heading together with its text
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The TF-IDF retrieval trials demonstrated consistent, query-relevant document ranking across "
        "the tested weighting-scheme configurations."
    )
    pdf.multi_cell(0, 5, notes_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Sign-off line
    if pdf.get_y() > pdf.h - 45:  # keep the line and its caption on the same page
        pdf.add_page()
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())
