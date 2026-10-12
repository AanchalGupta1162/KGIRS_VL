"""The References page: a simple grouped citation list."""
import streamlit as st

from ..config import REFERENCES

def render_references_section():
    """Renders the References page."""
    st.header("References", icon=":material/library_books:")
    

    total = sum(len(items) for _, _, items in REFERENCES)
    st.markdown(f":gray-badge[:material/format_list_numbered: {total} sources]")

    n = 0
    for group, icon, items in REFERENCES:
        st.subheader(group, icon=icon)
        for item in items:
            n += 1
            with st.container(border=True):
                st.markdown(f"**{n}.** &nbsp; {item['cite']}")
                link = item.get("url") or (f"https://doi.org/{item['doi']}" if item.get("doi") else None)
                if link:
                    label = f"doi:{item['doi']}" if item.get("doi") else "Read online"
                    st.markdown(f"[:material/link: {label}]({link})")
