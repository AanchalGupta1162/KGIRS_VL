"""The Theory page: six illustrated topics, then a glossary of key terms."""
import pandas as pd
import streamlit as st

from ..config import THEORY_TOPICS, THEORY_CONTENT
from ..graphics import _cosine_svg

# Width of the floated picture. Large enough to read comfortably; the body text wraps around it.
_IMAGE_WIDTH = 360


def _floated_topic_html(topic: dict) -> str:
    """Builds one topic's picture + body as a single markdown/HTML blob, with the picture floated
    left so the body text wraps around it instead of sitting in its own narrow column."""
    width = topic.get("image_width") or _IMAGE_WIDTH
    if topic["image"] == "svg:cosine":
        picture_html = _cosine_svg().replace(
            "<svg ", '<svg style="width:100%;height:auto;display:block;" ', 1
        )
    else:
        picture_html = (f'<img src="{topic["image"]}" alt="{topic["title"]}" '
                        f'style="width:100%;border-radius:8px;display:block;">')
    credit_html = (f'<div style="font-size:0.78rem;opacity:0.6;margin-top:0.35rem;">'
                   f'\U0001F4F7 {topic["credit"]}</div>') if topic["credit"] else ""

    return (
        f'<div style="overflow:auto;">\n\n'
        f'<div style="float:left;width:{width}px;margin:0 1.5rem 1rem 0;">\n\n'
        f'{picture_html}\n\n{credit_html}\n\n'
        f'</div>\n\n'
        f'{topic["body"]}\n\n'
        f'</div>'
    )


def render_theory_section():
    """Renders the Theory page: six topics, each with a picture, then the glossary."""
    st.header("Theory", icon=":material/menu_book:")
    
               

    for i, topic in enumerate(THEORY_TOPICS):
        with st.container(border=True):
            st.subheader(f"{i + 1}. {topic['title']}", icon=topic["icon"])
            st.markdown(_floated_topic_html(topic), unsafe_allow_html=True)
            if topic.get("latex"):
                st.latex(topic["latex"])

    with st.expander(f"Key terms ({len(THEORY_CONTENT['key_terms'])})", icon=":material/dictionary:",
                     key="key_terms_expander"):
        var_df = pd.DataFrame(list(THEORY_CONTENT["key_terms"].items()), columns=["Term", "Definition"])
        with st.container(key="key_terms_table"):
            st.table(var_df)

