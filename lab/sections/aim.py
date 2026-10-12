"""The Purpose page: the aim statement, plus the real-world-applications block."""
import math

import streamlit as st

from ..config import EXPERIMENT_CONFIG, APPLICATIONS, APPLICATION_EXPLANATIONS, TF_SCHEMES, IDF_SCHEMES
from ..engine import run_retrieval
from ..graphics import (
    APP_VISUAL_CSS, _draw, _esc, _highlight_html, _web_search_hero_svg, _web_search_pipeline_svg,
    _hero_svg, _pipeline_svg, _mini_svg, _benefit_art, _weight_grid_svg,
)

def render_aim_section():
    """Renders the Aim page: the aim statement, the objectives and what the student hands in."""
    st.header("Purpose", icon=":material/flag:")
    

    with st.container(border=True):
        st.markdown(EXPERIMENT_CONFIG["aim"])

    render_applications_block()



def render_applications_block():
    """Renders the orientation block at the foot of the Aim page: what the lab does, why it is worth
    using, a tour of its pages, and where the same technique is used in practice - plus a live retrieval
    over the chosen application's own collection. Not a page of its own; called by render_aim_section()."""
    st.header("Real-world applications", icon=":material/public:")
    
    st.html(APP_VISUAL_CSS)

    # --- First concrete example --------------------------------------------------------------------
    example = APPLICATIONS[0]
    st.subheader("Web and site search", icon=":material/travel_explore:")
    with st.container(border=True):
        st.markdown(
            "A search engine treats every page as a **document** and what the visitor types as the "
            "**query**. For example, the query *reset my account password* is compared with every help-centre page."
        )
        st.markdown(
            "**How TF-IDF helps:** TF counts how often a query term appears on a page. IDF gives more weight "
            "to specific terms such as *reset* because they appear in fewer pages, while common terms such as "
            "*account* contribute less. Cosine similarity combines these weighted terms and ranks the pages "
            "most likely to answer the question first."
        )
        example_left, example_right = st.columns(2)
        with example_left:
            st.markdown(f"**Query**  \n`{example['query']}`")
        with example_right:
            st.markdown(f"**Example document D1**  \n{example['corpus'][0]}")

        _draw(_web_search_hero_svg())
        with st.expander("How web search retrieval works, stage by stage", icon=":material/account_tree:"):
            _draw(_web_search_pipeline_svg())
            st.caption(
                "How a site search engine indexes help-centre pages, balances term weights with IDF, "
                "and serves the most relevant article first."
            )

        st.subheader("Watch one run", icon=":material/play_circle:")
        names = [a["name"] for a in APPLICATIONS]
        chosen = st.segmented_control(
            "Application", options=names, default=names[0], key="app_scenario", required=True,
            label_visibility="collapsed"
        )
        app = next((a for a in APPLICATIONS if a["name"] == chosen), APPLICATIONS[0])
    
        # Normalized TF with standard IDF and cosine: the defaults a student meets in the Simulation.
        result = run_retrieval(app["corpus"], app["query"], TF_SCHEMES[1], IDF_SCHEMES[0], use_cosine=True)
        doc_ids = [f"D{i + 1}" for i in range(len(app["corpus"]))]
        matched = set(result["query_tokens"])
        order = result["ranking"]
        peak = max(result["scores"]) or 1.0
    
        st.markdown(f"**The query** &nbsp; :material/search: &nbsp; *{app['query']}*")
    
        # Query terms as chips, sized by how much each one can move the ranking.
        terms_by_idf = sorted(matched, key=lambda t: result["idf"][t], reverse=True)
        top_idf = max((result["idf"][t] for t in terms_by_idf), default=0.0) or 1.0
        chips = []
        for t in terms_by_idf:
            size = 13 + 13 * (result["idf"][t] / top_idf)
            chips.append(f'<span class="kg-chip" style="font-size:{size:.0f}px" title="idf {result["idf"][t]:.3f} '
                         f'· appears in {result["df"][t]} of {len(app["corpus"])} documents">{_esc(t)}</span>')
        for t in result["oov_terms"]:
            chips.append(f'<span class="kg-chip kg-dead" style="font-size:13px" '
                         f'title="not in this collection, ignored">{_esc(t)}</span>')
        if chips:
            st.html(f'<div class="kg-chips">{"".join(chips)}</div>')
            
                       
    
        st.markdown("**The ranking**")
        rows = []
        for rank, i in enumerate(order, start=1):
            score = result["scores"][i]
            pct = 100 * score / peak
            rows.append(
                f'<div class="kg-row">'
                f'<div class="kg-rank{" kg-top" if rank == 1 and score > 0 else ""}">{rank}</div>'
                f'<div class="kg-body">'
                f'<div class="kg-bar{"" if score > 0 else " kg-zero"}"><span style="width:{max(pct, 1.2):.1f}%"></span></div>'
                f'<div class="kg-text">{_highlight_html(app["corpus"][i], matched)}</div>'
                f'</div>'
                f'<div class="kg-score">{score:.3f}</div>'
                f'</div>'
            )
        st.html(f'<div class="kg-wrap">{"".join(rows)}</div>')
        st.info(app["why"], icon=":material/lightbulb:")
    
        if matched:
            with st.expander("The weights behind those bars", icon=":material/grid_on:"):
                terms = terms_by_idf
                weights = [[result["tfidf_matrix"][i][t] for t in terms] for i in order]
                _draw(_weight_grid_svg(terms, [doc_ids[i] for i in order], weights))
                st.caption("Rows in ranked order. Same numbers as the Simulation's Matrices tab.")
    
        st.divider()
    
    # --- What this lab does -------------------------------------------------------------------------
    st.subheader("What this lab does", icon=":material/lightbulb:")
    _draw(_hero_svg())
    with st.expander("How it works, stage by stage", icon=":material/account_tree:"):
        _draw(_pipeline_svg())
        st.caption("Only the collection and the query change between applications. The grey line under "
                   "each stage is where you watch it happen in the Simulation.")

    # --- Why use it ---------------------------------------------------------------------------------
    st.subheader("Why use it", icon=":material/star:")
    reasons = [
        ("Nothing is hidden", "Every TF, IDF and TF-IDF number that produced the ranking is on screen."),
        ("Change one setting", "Switch a weighting scheme and watch the order of the results move."),
        ("Judge the ranking", "Score it with the same parameters real search systems are judged on."),
        ("Leave with a report", "Your trials and observations export as a PDF you can hand in."),
    ]
    art = _benefit_art()
    cols = st.columns(4)
    for col, (title, text), drawing in zip(cols, reasons, art):
        with col:
            with st.container(border=True):
                _draw(_mini_svg(drawing, title), width=120)
                st.markdown(f"**{title}**")
                st.caption(text)

    st.divider()

    # --- Gallery ------------------------------------------------------------------------------------
    st.subheader("Where it is used", icon=":material/apps:")
    for row_start in range(0, len(APPLICATIONS), 3):
        cols = st.columns(3)
        for col, a in zip(cols, APPLICATIONS[row_start:row_start + 3]):
            with col:
                with st.expander(f"{a['icon']}  {a['name']}", expanded=False):
                    st.markdown(
                        f":gray-badge[:material/description: {a['doc']}]  \n"
                        f":gray-badge[:material/search: {a['query_is']}]  \n"
                        f":blue-badge[:material/straighten: {a['metric']}]"
                    )
                    st.markdown(APPLICATION_EXPLANATIONS[a["name"]])

    st.divider()

    # --- Live retrieval over the chosen application -------------------------------------------------
    

    # --- Why a rare term decides the ranking --------------------------------------------------------
    st.subheader("Why the rare word wins", icon=":material/insights:")
    st.markdown("IDF is $\\log_{10}(N / \\mathrm{df}_t)$ - it collapses as a term spreads.")
    idf_rows = []
    for df_val in (1, 2, 5, 20, 50, 100):
        idf_val = math.log10(100 / df_val)
        idf_rows.append(
            f'<div class="kg-idfrow">'
            f'<div class="kg-idflab">in {df_val} of 100 documents</div>'
            f'<div class="kg-bar"><span style="width:{max(100 * idf_val / 2.0, 0.6):.1f}%"></span></div>'
            f'<div class="kg-idfval">{idf_val:.2f}</div>'
            f'</div>'
        )
    st.html(f'<div class="kg-wrap">{"".join(idf_rows)}</div>')