"""Hand-drawn SVG illustrations and small markup helpers shared across pages.

Kept separate from the page renderers so a page module only pulls in the drawings it
actually uses, and so the drawing code (verbose, mostly numeric) doesn't crowd out the
page logic that calls it.
"""
import math
import re
from html import escape as _html_escape

import streamlit as st

from .config import PIPELINE_STAGES
from .engine import tokenize

def _cosine_svg() -> str:
    """Query and two document vectors, drawn to show that the smaller angle is the better match.

    Commons has no clear cosine-similarity illustration, so this topic is drawn rather than photographed.
    """
    ox, oy = 62, 252
    p = [_svg_open(430, 300, "Query and document vectors, ranked by the angle between them")]
    p.append(f'<line x1="{ox}" y1="{oy}" x2="410" y2="{oy}" stroke="currentColor" stroke-opacity="0.35" '
             f'stroke-width="2"/>')
    p.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="36" stroke="currentColor" stroke-opacity="0.35" '
             f'stroke-width="2"/>')
    p.append(f'<text x="404" y="{oy + 22}" text-anchor="end" font-size="12" fill="currentColor" '
             f'fill-opacity="0.55">weight of term A</text>')
    p.append(f'<text x="{ox - 8}" y="44" text-anchor="end" font-size="12" fill="currentColor" '
             f'fill-opacity="0.55">term B</text>')

    vectors = [((300, 96), ACCENT, 3.4, "q", "query"),
               ((330, 150), "currentColor", 2.6, "d1", "best match"),
               ((158, 58), "currentColor", 2.6, "d2", "poor match")]
    for (ex, ey), color, w, label, note in vectors:
        op = "1" if color == ACCENT else "0.55"
        p.append(f'<line x1="{ox}" y1="{oy}" x2="{ex}" y2="{ey}" stroke="{color}" stroke-opacity="{op}" '
                 f'stroke-width="{w}" stroke-linecap="round"/>')
        p.append(f'<circle cx="{ex}" cy="{ey}" r="5" fill="{color}" fill-opacity="{op}"/>')
        p.append(f'<text x="{ex + 12}" y="{ey - 2}" font-size="15" font-weight="700" fill="{color}" '
                 f'fill-opacity="{op}">{label}</text>')
        p.append(f'<text x="{ex + 12}" y="{ey + 15}" font-size="11" fill="currentColor" '
                 f'fill-opacity="0.55">{note}</text>')

    # The angle between q and the better-matching document, drawn as a shaded wedge so it reads
    # clearly even though the two vectors are, by design, close together (that is what makes d1
    # the better match). SVG angles run clockwise from the +x axis since y points down the page.
    r = 70
    q_angle = math.degrees(math.atan2(96 - oy, 300 - ox))
    d1_angle = math.degrees(math.atan2(150 - oy, 330 - ox))
    start_angle, end_angle = sorted((q_angle, d1_angle))

    def _point_on_circle(angle_deg, radius):
        rad = math.radians(angle_deg)
        return ox + radius * math.cos(rad), oy + radius * math.sin(rad)

    sx, sy = _point_on_circle(start_angle, r)
    ex2, ey2 = _point_on_circle(end_angle, r)
    large_arc = 1 if (end_angle - start_angle) > 180 else 0
    p.append(f'<path d="M {ox} {oy} L {sx:.1f} {sy:.1f} A {r} {r} 0 {large_arc} 1 {ex2:.1f} {ey2:.1f} Z" '
             f'fill="{ACCENT}" fill-opacity="0.14" stroke="{ACCENT}" stroke-width="1.5" '
             f'stroke-opacity="0.8"/>')
    label_angle = (start_angle + end_angle) / 2
    lx, ly = _point_on_circle(label_angle, r + 20)
    p.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" '
             f'font-size="14" fill="{ACCENT}">&#952;</text>')
    p.append(f'<text x="{ox + 4}" y="{oy + 22}" font-size="11" fill="currentColor" fill-opacity="0.55">'
             f'origin</text>')
    p.append("</svg>")
    return "".join(p)


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+!|~$:])")


def _md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def highlight_terms(text: str, terms: set) -> str:
    """Returns markdown for `text` with every word whose token is a query term highlighted."""
    parts = re.split(r"(\s+)", text)
    out = []
    for part in parts:
        if part.strip() and set(tokenize(part)) & terms:
            out.append(f":orange-background[**{_md_escape(part)}**]")
        else:
            out.append(_md_escape(part))
    return "".join(out)


ACCENT = "#2A9AA4"


APP_VISUAL_CSS = """
<style>
.kg-wrap { width: 100%; }
.kg-chips { display: flex; flex-wrap: wrap; align-items: baseline; gap: .35rem; }
.kg-chip {
    display: inline-block; padding: .15em .6em; border-radius: 999px; font-weight: 600;
    background: rgba(42, 154, 164, 0.16); border: 1px solid rgba(42, 154, 164, 0.45); line-height: 1.5;
}
.kg-chip.kg-dead {
    background: rgba(128, 128, 128, 0.10); border-color: rgba(128, 128, 128, 0.30); opacity: .65;
}
.kg-row { display: flex; align-items: center; gap: .75rem; padding: .45rem 0;
          border-bottom: 1px solid rgba(128, 128, 128, 0.22); }
.kg-row:last-child { border-bottom: none; }
.kg-rank { flex: 0 0 1.9rem; height: 1.9rem; border-radius: 50%; display: flex; align-items: center;
           justify-content: center; font-weight: 700; font-size: 1.05rem;
           background: rgba(128, 128, 128, 0.18); }
.kg-rank.kg-top { background: #2A9AA4; color: #fff; }
.kg-body { flex: 1 1 auto; min-width: 0; }
.kg-bar { height: 8px; border-radius: 4px; background: rgba(128, 128, 128, 0.16); overflow: hidden;
          margin-bottom: .3rem; }
.kg-bar > span { display: block; height: 100%; border-radius: 4px; background: #2A9AA4; }
.kg-bar.kg-zero > span { background: rgba(128, 128, 128, 0.35); }
.kg-text { font-size: 1.1rem; line-height: 1.5; opacity: .92; }
.kg-score { flex: 0 0 3.4rem; text-align: right; font-variant-numeric: tabular-nums;
            font-weight: 600; font-size: 1.05rem; }
.kg-hit { background: rgba(224, 146, 58, 0.30); color: inherit; padding: 0 .18em; border-radius: 3px; }
.kg-idfrow { display: flex; align-items: center; gap: .9rem; padding: .4rem 0; }
.kg-idflab { flex: 0 0 12rem; font-size: 1.05rem; opacity: .75; }
.kg-idfval { flex: 0 0 2.6rem; text-align: right; font-variant-numeric: tabular-nums;
             font-weight: 600; font-size: 1.05rem; }
</style>
"""


def _esc(text) -> str:
    return _html_escape(str(text), quote=True)


def _highlight_html(text: str, terms: set) -> str:
    """Same idea as highlight_terms(), but emitting HTML for the illustrated result rows."""
    out = []
    for part in re.split(r"(\s+)", text):
        if part.strip() and set(tokenize(part)) & terms:
            out.append(f'<mark class="kg-hit">{_esc(part)}</mark>')
        else:
            out.append(_esc(part))
    return "".join(out)



def _pipeline_svg() -> str:
    """Draws the six retrieval stages as a single illustrated flow strip.

    Every shape uses currentColor or the accent, so the drawing follows the active theme.
    """
    art = []
    for i in range(len(PIPELINE_STAGES)):
        cx = i * 200 + 80
        if i == 0:  # a small shelf of documents
            shapes = []
            for j in range(3):
                dx = cx - 46 + j * 32
                shapes.append(f'<rect x="{dx}" y="56" width="28" height="42" rx="3" fill="currentColor" '
                              f'fill-opacity="0.10" stroke="currentColor" stroke-opacity="0.38"/>')
                for ly in (66, 74, 82):
                    shapes.append(f'<line x1="{dx + 5}" y1="{ly}" x2="{dx + 23}" y2="{ly}" '
                                  f'stroke="currentColor" stroke-opacity="0.38" stroke-width="2"/>')
            art.append("".join(shapes))
        elif i == 1:  # a document broken into loose tokens
            shapes = [f'<rect x="{cx - 58}" y="58" width="26" height="40" rx="3" fill="currentColor" '
                      f'fill-opacity="0.10" stroke="currentColor" stroke-opacity="0.38"/>']
            for ly in (68, 76, 84):
                shapes.append(f'<line x1="{cx - 53}" y1="{ly}" x2="{cx - 37}" y2="{ly}" '
                              f'stroke="currentColor" stroke-opacity="0.38" stroke-width="2"/>')
            for r in range(2):
                for c in range(3):
                    shapes.append(f'<rect x="{cx - 20 + c * 24}" y="{60 + r * 22}" width="18" height="12" '
                                  f'rx="3" fill="{ACCENT}" fill-opacity="0.42"/>')
            art.append("".join(shapes))
        elif i == 2:  # a weight grid with uneven cells
            ops = [0.14, 0.55, 0.22, 0.80, 0.38, 0.10, 0.66, 0.26, 0.18, 0.44, 0.90, 0.30]
            shapes = []
            for idx, op in enumerate(ops):
                r, c = divmod(idx, 4)
                shapes.append(f'<rect x="{cx - 40 + c * 21}" y="{54 + r * 21}" width="17" height="17" '
                              f'rx="3" fill="{ACCENT}" fill-opacity="{op}"/>')
            art.append("".join(shapes))
        elif i == 3:  # vectors leaving the origin
            shapes = [f'<line x1="{cx - 34}" y1="108" x2="{cx + 42}" y2="108" stroke="currentColor" '
                      f'stroke-opacity="0.35" stroke-width="2"/>',
                      f'<line x1="{cx - 34}" y1="108" x2="{cx - 34}" y2="52" stroke="currentColor" '
                      f'stroke-opacity="0.35" stroke-width="2"/>']
            for ex, ey, op in ((cx + 22, 62, 0.9), (cx + 36, 84, 0.55), (cx - 4, 56, 0.55)):
                shapes.append(f'<line x1="{cx - 34}" y1="108" x2="{ex}" y2="{ey}" stroke="{ACCENT}" '
                              f'stroke-opacity="{op}" stroke-width="2.5" marker-end="url(#kgtip)"/>')
            art.append("".join(shapes))
        elif i == 4:  # the angle between query and document
            art.append(
                f'<line x1="{cx - 30}" y1="108" x2="{cx + 42}" y2="108" stroke="currentColor" '
                f'stroke-opacity="0.35" stroke-width="2"/>'
                f'<line x1="{cx - 30}" y1="108" x2="{cx - 30}" y2="52" stroke="currentColor" '
                f'stroke-opacity="0.35" stroke-width="2"/>'
                f'<line x1="{cx - 30}" y1="108" x2="{cx + 34}" y2="54" stroke="{ACCENT}" stroke-width="2.8" '
                f'marker-end="url(#kgtip)"/>'
                f'<line x1="{cx - 30}" y1="108" x2="{cx + 40}" y2="84" stroke="currentColor" '
                f'stroke-opacity="0.6" stroke-width="2.5" marker-end="url(#kgtip)"/>'
                f'<path d="M {cx - 4} 86 A 34 34 0 0 1 {cx + 3} 97" fill="none" stroke="currentColor" '
                f'stroke-opacity="0.65" stroke-width="1.6"/>'
                f'<text x="{cx + 9}" y="90" font-size="12" fill="currentColor" fill-opacity="0.75">&#952;</text>'
            )
        else:  # the ranked list
            shapes = []
            for j, w in enumerate((92, 68, 46, 28)):
                fill = ACCENT if j == 0 else "currentColor"
                op = "0.95" if j == 0 else "0.28"
                shapes.append(f'<rect x="{cx - 46}" y="{56 + j * 16}" width="{w}" height="11" rx="3" '
                              f'fill="{fill}" fill-opacity="{op}"/>')
            art.append("".join(shapes))

    parts = [
        _svg_open(1160, 186, "The six stages of the TF-IDF retrieval pipeline"),
        '<defs><marker id="kgtip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>',
    ]
    for i, stage in enumerate(PIPELINE_STAGES):
        x, cx = i * 200, i * 200 + 80
        parts.append(f'<rect x="{x}" y="20" width="160" height="130" rx="14" fill="currentColor" '
                     f'fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.22"/>')
        parts.append(art[i])
        parts.append(f'<circle cx="{x + 21}" cy="35" r="12" fill="{ACCENT}"/>')
        parts.append(f'<text x="{x + 21}" y="40" text-anchor="middle" font-size="13" font-weight="700" '
                     f'fill="#ffffff">{stage["n"]}</text>')
        parts.append(f'<text x="{cx}" y="137" text-anchor="middle" font-size="14" font-weight="600" '
                     f'fill="currentColor">{_esc(stage["name"])}</text>')
        parts.append(f'<text x="{cx}" y="172" text-anchor="middle" font-size="11" fill="currentColor" '
                     f'fill-opacity="0.55">{_esc(stage["where"])}</text>')
        if i < len(PIPELINE_STAGES) - 1:
            parts.append(f'<line x1="{x + 168}" y1="85" x2="{x + 190}" y2="85" stroke="currentColor" '
                         f'stroke-opacity="0.45" stroke-width="2" marker-end="url(#kgtip)"/>')
    parts.append("</svg>")
    return "".join(parts)




def _weight_grid_svg(terms: list, doc_ids: list, weights: list) -> str:
    """Term-by-document TF-IDF weights drawn as a shaded grid: darker cell, heavier weight."""
    cell_w, cell_h, gutter_l, gutter_t = 78, 30, 46, 26
    width = gutter_l + len(terms) * cell_w
    height = gutter_t + len(doc_ids) * cell_h
    peak = max((w for row in weights for w in row), default=0.0) or 1.0

    parts = [_svg_open(width, height, "TF-IDF weight of each query term in each document")]
    for c, term in enumerate(terms):
        parts.append(f'<text x="{gutter_l + c * cell_w + cell_w / 2}" y="17" text-anchor="middle" '
                     f'font-size="13" font-weight="600" fill="currentColor">{_esc(term)}</text>')
    for r, doc in enumerate(doc_ids):
        y = gutter_t + r * cell_h
        parts.append(f'<text x="{gutter_l - 10}" y="{y + cell_h / 2 + 4}" text-anchor="end" font-size="12" '
                     f'fill="currentColor" fill-opacity="0.7">{_esc(doc)}</text>')
        for c in range(len(terms)):
            w = weights[r][c]
            op = round(max(0.05, w / peak), 3)
            x = gutter_l + c * cell_w
            parts.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" '
                         f'fill="{ACCENT}" fill-opacity="{op}"><title>{_esc(doc)} · {_esc(terms[c])} '
                         f'= {w:.4f}</title></rect>')
            if w > 0:
                parts.append(f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" text-anchor="middle" '
                             f'font-size="11" fill="currentColor" fill-opacity="0.9">{w:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)




def _svg_open(view_w: int, view_h: int, label: str) -> str:
    # A drawing rendered through st.image is an isolated document and cannot reach the page's web fonts,
    # so name a system sans-serif stack here instead of inheriting the serif default.
    return (f'<svg viewBox="0 0 {view_w} {view_h}" role="img" aria-label="{_esc(label)}" '
            f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">')

def _draw(svg: str, width="stretch") -> None:
    """Renders SVG using inline CSS for dynamic theme switching."""
    st.image(svg, width=width)



def _hero_svg() -> str:
    """The one-drawing answer to 'what does this website actually do?'.

    Three scenes - you ask, it weighs every word, you get a ranking you can measure.
    """
    p = [_svg_open(1100, 232, "You ask a question, the lab weighs every word, you get a measured ranking"),
         '<defs><marker id="kghero" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
         'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>']

    # Scene A - the query box
    p.append('<rect x="34" y="66" width="250" height="52" rx="26" fill="currentColor" fill-opacity="0.05" '
             'stroke="currentColor" stroke-opacity="0.30"/>')
    p.append(f'<circle cx="68" cy="92" r="12" fill="none" stroke="{ACCENT}" stroke-width="3"/>')
    p.append(f'<line x1="77" y1="101" x2="88" y2="112" stroke="{ACCENT}" stroke-width="3.5" '
             f'stroke-linecap="round"/>')
    p.append('<text x="104" y="98" font-size="15" fill="currentColor" fill-opacity="0.75">'
             'tf-idf document ranking</text>')

    # Scene B - documents fanned into a weight matrix
    for j in range(3):
        dx = 372 + j * 26
        p.append(f'<rect x="{dx}" y="58" width="22" height="30" rx="3" fill="currentColor" '
                 f'fill-opacity="0.10" stroke="currentColor" stroke-opacity="0.35"/>')
    p.append('<text x="411" y="106" text-anchor="middle" font-size="11" fill="currentColor" '
             'fill-opacity="0.55">documents</text>')
    p.append(f'<line x1="452" y1="80" x2="474" y2="80" stroke="currentColor" stroke-opacity="0.4" '
             f'stroke-width="2" marker-end="url(#kghero)"/>')
    ops = [0.14, 0.62, 0.22, 0.85, 0.30,
           0.40, 0.12, 0.70, 0.20, 0.55,
           0.25, 0.48, 0.16, 0.36, 0.92,
           0.60, 0.18, 0.30, 0.74, 0.22]
    for idx, op in enumerate(ops):
        r, c = divmod(idx, 5)
        p.append(f'<rect x="{492 + c * 30}" y="{44 + r * 26}" width="26" height="22" rx="4" '
                 f'fill="{ACCENT}" fill-opacity="{op}"/>')
    p.append('<text x="561" y="164" text-anchor="middle" font-size="11" fill="currentColor" '
             'fill-opacity="0.55">every word, weighted</text>')

    # Scene C - the ranked answer
    for j, w in enumerate((156, 112, 76, 48)):
        top = j == 0
        p.append(f'<text x="852" y="{62 + j * 24}" text-anchor="end" font-size="12" fill="currentColor" '
                 f'fill-opacity="0.6">{j + 1}</text>')
        p.append(f'<rect x="862" y="{50 + j * 24}" width="{w}" height="16" rx="4" '
                 f'fill="{ACCENT if top else "currentColor"}" fill-opacity="{1 if top else 0.24}"/>')
    p.append(f'<circle cx="1038" cy="58" r="11" fill="{ACCENT}"/>')
    p.append('<path d="M 1032 58 L 1036 63 L 1044 53" fill="none" stroke="#ffffff" stroke-width="2.6" '
             'stroke-linecap="round" stroke-linejoin="round"/>')
    p.append('<text x="862" y="164" font-size="11" fill="currentColor" fill-opacity="0.55">'
             'P@k · R@k · F1 · AP · nDCG</text>')

    # Connectors and captions
    p.append('<line x1="300" y1="92" x2="358" y2="92" stroke="currentColor" stroke-opacity="0.45" '
             'stroke-width="2.5" marker-end="url(#kghero)"/>')
    p.append('<line x1="660" y1="92" x2="846" y2="92" stroke="currentColor" stroke-opacity="0.45" '
             'stroke-width="2.5" marker-end="url(#kghero)"/>')
    for cx, title, sub in ((159, "You ask", "any query, any collection"),
                           (561, "It weighs every word", "rare words count, common ones do not"),
                           (940, "You get a ranking", "and the numbers that judge it")):
        p.append(f'<text x="{cx}" y="198" text-anchor="middle" font-size="16" font-weight="700" '
                 f'fill="currentColor">{_esc(title)}</text>')
        p.append(f'<text x="{cx}" y="219" text-anchor="middle" font-size="12" fill="currentColor" '
                 f'fill-opacity="0.6">{_esc(sub)}</text>')
    p.append("</svg>")
    return "".join(p)




def _mini_svg(inner: str, label: str) -> str:
    return _svg_open(120, 80, label) + inner + "</svg>"




def _benefit_art() -> list:
    """Four small drawings, one per reason to use the lab."""
    # 1. every number is on screen: a weight matrix under a magnifier
    grid = []
    for idx, op in enumerate([0.15, 0.55, 0.25, 0.80, 0.35, 0.12, 0.65, 0.30, 0.20, 0.45, 0.90, 0.28]):
        r, c = divmod(idx, 4)
        grid.append(f'<rect x="{14 + c * 21}" y="{8 + r * 18}" width="18" height="15" rx="3" '
                    f'fill="{ACCENT}" fill-opacity="{op}"/>')
    grid.append(f'<circle cx="88" cy="56" r="14" fill="none" stroke="{ACCENT}" stroke-width="3.2"/>')
    grid.append(f'<line x1="98" y1="66" x2="110" y2="78" stroke="{ACCENT}" stroke-width="3.5" '
                f'stroke-linecap="round"/>')

    # 2. one switch, a different ranking
    knob = [f'<rect x="12" y="8" width="46" height="24" rx="12" fill="{ACCENT}" fill-opacity="0.22" '
            f'stroke="{ACCENT}" stroke-opacity="0.6"/>',
            f'<circle cx="46" cy="20" r="9" fill="{ACCENT}"/>']
    for j, w in enumerate((32, 21, 13)):
        knob.append(f'<rect x="8" y="{44 + j * 12}" width="{w}" height="8" rx="3" fill="currentColor" '
                    f'fill-opacity="0.28"/>')
    # Drawn arrowhead rather than a marker: markers live in another SVG's <defs> and may not be rendered.
    knob.append('<line x1="50" y1="56" x2="60" y2="56" stroke="currentColor" stroke-opacity="0.45" '
                'stroke-width="2"/>')
    knob.append('<polygon points="60,52 67,56 60,60" fill="currentColor" fill-opacity="0.45"/>')
    for j, (w, hot) in enumerate(((21, False), (32, True), (11, False))):
        knob.append(f'<rect x="70" y="{44 + j * 12}" width="{w}" height="8" rx="3" '
                    f'fill="{ACCENT if hot else "currentColor"}" fill-opacity="{1 if hot else 0.28}"/>')

    # 3. a gauge, because the ranking gets scored
    gauge = ['<path d="M 22 62 A 38 38 0 0 1 98 62" fill="none" stroke="currentColor" stroke-opacity="0.18" '
             'stroke-width="11" stroke-linecap="round"/>',
             f'<path d="M 22 62 A 38 38 0 0 1 98 62" fill="none" stroke="{ACCENT}" stroke-width="11" '
             f'stroke-linecap="round" stroke-dasharray="84 126"/>',
             '<line x1="60" y1="62" x2="79" y2="36" stroke="currentColor" stroke-width="3" '
             'stroke-linecap="round"/>',
             '<circle cx="60" cy="62" r="4.5" fill="currentColor"/>',
             '<text x="60" y="78" text-anchor="middle" font-size="12" font-weight="700" '
             'fill="currentColor">nDCG</text>']

    # 4. you leave with a PDF
    report = ['<rect x="22" y="8" width="54" height="64" rx="5" fill="currentColor" fill-opacity="0.07" '
              'stroke="currentColor" stroke-opacity="0.35"/>']
    for j, w in enumerate((36, 36, 26, 36, 20)):
        report.append(f'<rect x="30" y="{18 + j * 10}" width="{w}" height="4" rx="2" fill="currentColor" '
                      f'fill-opacity="0.3"/>')
    report.append(f'<circle cx="90" cy="56" r="16" fill="{ACCENT}"/>')
    report.append('<line x1="90" y1="47" x2="90" y2="63" stroke="#ffffff" stroke-width="3" '
                  'stroke-linecap="round"/>')
    report.append('<path d="M 83 56 L 90 64 L 97 56" fill="none" stroke="#ffffff" stroke-width="3" '
                  'stroke-linecap="round" stroke-linejoin="round"/>')

    return ["".join(grid), "".join(knob), "".join(gauge), "".join(report)]



def _web_search_hero_svg() -> str:
    """An illustrated 3-scene diagram demonstrating web and site search with TF-IDF."""
    p = [
        _svg_open(1100, 250, "How web and site search retrieves and ranks help-centre pages using TF-IDF"),
        f'<defs>'
        f'<marker id="kgwebflow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" class="svg-ink"/></marker>'
        f'<style>'
        f'  .svg-ink {{ fill: #1E293B; stroke: none; }}'
        f'  .svg-stroke {{ stroke: #1E293B; }}'
        f'  .svg-card-bg {{ fill: #F8FAFC; stroke: #CBD5E1; }}'
        f'  .svg-subcard-bg {{ fill: #F1F5F9; stroke: #E2E8F0; }}'
        f'  .svg-text-muted {{ fill: #64748B; }}'
        f'  @media (prefers-color-scheme: dark) {{'
        f'    .svg-ink {{ fill: #F8FAFC; }}'
        f'    .svg-stroke {{ stroke: #F8FAFC; }}'
        f'    .svg-card-bg {{ fill: rgba(255, 255, 255, 0.05); stroke: rgba(255, 255, 255, 0.2); }}'
        f'    .svg-subcard-bg {{ fill: rgba(255, 255, 255, 0.08); stroke: rgba(255, 255, 255, 0.15); }}'
        f'    .svg-text-muted {{ fill: #94A3B8; }}'
        f'  }}'
        f'</style>'
        f'</defs>'
    ]

    # --- Scene 1: User Search Query & Parsing (Left: x=30 to x=320) ---
    p.append('<rect x="30" y="20" width="280" height="162" rx="10" class="svg-card-bg" stroke-width="1.4"/>')
    p.append(f'<circle cx="48" cy="36" r="3.5" fill="{ACCENT}"/>')
    p.append('<circle cx="59" cy="36" r="3.5" class="svg-text-muted"/>')
    p.append('<circle cx="70" cy="36" r="3.5" class="svg-text-muted"/>')
    p.append('<rect x="84" y="27" width="210" height="18" rx="4" class="svg-subcard-bg"/>')
    p.append('<text x="94" y="40" font-size="9.5" font-weight="600" class="svg-ink">help.company.com/search</text>')

    p.append(f'<rect x="44" y="58" width="252" height="36" rx="18" class="svg-subcard-bg" stroke="{ACCENT}" stroke-width="2"/>')
    p.append(f'<circle cx="62" cy="76" r="6.5" fill="none" stroke="{ACCENT}" stroke-width="2.2"/>')
    p.append(f'<line x1="67" y1="81" x2="74" y2="88" stroke="{ACCENT}" stroke-width="2.4" stroke-linecap="round"/>')
    p.append('<text x="82" y="80" font-size="12" font-weight="700" class="svg-ink">reset my account password</text>')

    p.append(f'<rect x="44" y="106" width="56" height="22" rx="11" fill="{ACCENT}" fill-opacity="0.25" stroke="{ACCENT}" stroke-width="1.4"/>')
    p.append(f'<text x="72" y="121" text-anchor="middle" font-size="11" font-weight="700" fill="{ACCENT}">reset</text>')

    p.append('<rect x="106" y="106" width="68" height="22" rx="11" class="svg-subcard-bg"/>')
    p.append('<text x="140" y="121" text-anchor="middle" font-size="11" font-weight="600" class="svg-ink">account</text>')

    p.append('<rect x="180" y="106" width="76" height="22" rx="11" class="svg-subcard-bg"/>')
    p.append('<text x="218" y="121" text-anchor="middle" font-size="11" font-weight="600" class="svg-ink">password</text>')

    p.append('<text x="170" y="152" text-anchor="middle" font-size="10" font-style="italic" class="svg-text-muted">(\'my\' dropped as stop word)</text>')

    # --- Scene 2: Corpus TF-IDF Weighting (Center: x=365 to x=725) ---
    p.append('<rect x="365" y="20" width="355" height="162" rx="10" class="svg-card-bg" stroke-width="1.4"/>')
    p.append('<text x="382" y="42" font-size="12" font-weight="700" class="svg-ink">Help-Centre Corpus (5 pages)</text>')
    p.append(f'<text x="702" y="42" text-anchor="end" font-size="11" font-weight="700" fill="{ACCENT}">IDF Weight</text>')

    # reset (rare -> high IDF)
    p.append(f'<rect x="380" y="54" width="325" height="26" rx="5" fill="{ACCENT}" fill-opacity="0.2" stroke="{ACCENT}" stroke-width="1.2"/>')
    p.append(f'<text x="390" y="71" font-size="11.5" font-weight="700" fill="{ACCENT}">reset</text>')
    p.append('<text x="446" y="71" font-size="10" font-weight="600" class="svg-ink">in 2 of 5 docs (rare)</text>')
    p.append('<rect x="555" y="60" width="105" height="14" rx="3" class="svg-subcard-bg"/>')
    p.append(f'<rect x="555" y="60" width="90" height="14" rx="3" fill="{ACCENT}"/>')
    p.append(f'<text x="670" y="72" font-size="11.5" font-weight="800" fill="{ACCENT}">0.40</text>')
    p.append(f'<circle cx="696" cy="67" r="6.5" fill="{ACCENT}"/>')
    p.append('<path d="M 693 67 L 695 70 L 699 64" fill="none" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round"/>')

    # password (medium)
    p.append('<text x="390" y="99" font-size="11.5" font-weight="700" class="svg-ink">password</text>')
    p.append('<text x="446" y="99" font-size="10" font-weight="600" class="svg-ink">in 3 of 5 docs (medium)</text>')
    p.append('<rect x="555" y="88" width="105" height="14" rx="3" class="svg-subcard-bg"/>')
    p.append(f'<rect x="555" y="88" width="50" height="14" rx="3" fill="{ACCENT}" fill-opacity="0.75"/>')
    p.append('<text x="670" y="100" font-size="11.5" font-weight="700" class="svg-ink">0.22</text>')

    # account (common -> low IDF)
    p.append('<text x="390" y="126" font-size="11.5" font-weight="600" class="svg-ink">account</text>')
    p.append('<text x="446" y="126" font-size="10" font-weight="600" class="svg-ink">in 4 of 5 docs (common)</text>')
    p.append('<rect x="555" y="115" width="105" height="14" rx="3" class="svg-subcard-bg"/>')
    p.append('<rect x="555" y="115" width="22" height="14" rx="3" fill="#94A3B8"/>')
    p.append('<text x="670" y="127" font-size="11.5" font-weight="700" class="svg-ink">0.10</text>')

    # banner
    p.append(f'<rect x="380" y="145" width="325" height="23" rx="4" fill="{ACCENT}" fill-opacity="0.18" stroke="{ACCENT}" stroke-width="1"/>')
    p.append(f'<text x="542" y="161" text-anchor="middle" font-size="10.5" font-weight="700" fill="{ACCENT}">Rare term "reset" carries 4x the discriminative weight of "account"</text>')

    # --- Scene 3: Ranked SERP Results (Right: x=770 to x=1070) ---
    p.append('<rect x="770" y="20" width="300" height="162" rx="10" class="svg-card-bg" stroke-width="1.4"/>')
    p.append('<text x="786" y="40" font-size="12" font-weight="700" class="svg-ink">Ranked Search Results (SERP)</text>')
    p.append('<text x="1054" y="40" text-anchor="end" font-size="11" font-weight="700" class="svg-ink">Score</text>')

    # Result #1: D1 (Winner)
    p.append(f'<rect x="782" y="48" width="276" height="38" rx="5" fill="{ACCENT}" fill-opacity="0.22" stroke="{ACCENT}" stroke-width="1.8"/>')
    p.append(f'<rect x="788" y="55" width="18" height="18" rx="3" fill="{ACCENT}"/>')
    p.append('<text x="797" y="68" text-anchor="middle" font-size="11" font-weight="800" fill="#FFFFFF">1</text>')
    p.append('<text x="814" y="63" font-size="11" font-weight="700" class="svg-ink">D1: Reset your password...</text>')
    p.append('<text x="814" y="78" font-size="9.5" font-weight="600" class="svg-text-muted">Account settings &gt; link emailed</text>')
    p.append(f'<text x="1048" y="70" text-anchor="end" font-size="12" font-weight="800" fill="{ACCENT}">0.323</text>')

    # Result #2: D4
    p.append('<rect x="782" y="90" width="276" height="30" rx="5" class="svg-subcard-bg"/>')
    p.append('<rect x="788" y="96" width="16" height="16" rx="3" fill="#94A3B8"/>')
    p.append('<text x="796" y="108" text-anchor="middle" font-size="10" font-weight="700" fill="#FFFFFF">2</text>')
    p.append('<text x="814" y="104" font-size="10.5" font-weight="700" class="svg-ink">D4: Support desk sign in help</text>')
    p.append('<text x="814" y="115" font-size="8.5" font-weight="600" class="svg-text-muted">...after a password reset</text>')
    p.append('<text x="1048" y="109" text-anchor="end" font-size="11" font-weight="700" class="svg-ink">0.244</text>')

    # Result #3: D3
    p.append('<rect x="782" y="124" width="276" height="24" rx="4" class="svg-subcard-bg"/>')
    p.append('<text x="796" y="140" text-anchor="middle" font-size="10" font-weight="700" class="svg-ink">3</text>')
    p.append('<text x="814" y="139" font-size="9.5" font-weight="600" class="svg-ink">D3: Password policy requirements</text>')
    p.append('<text x="1048" y="140" text-anchor="end" font-size="10.5" font-weight="600" class="svg-ink">0.052</text>')

    p.append('<text x="786" y="167" font-size="9.5" font-weight="600" class="svg-text-muted">D2 (billing): 0.010 &#160;·&#160; D5 (company info): 0.000</text>')

    # Connecting arrows
    p.append('<line x1="318" y1="100" x2="355" y2="100" class="svg-stroke" stroke-width="2.6" marker-end="url(#kgwebflow)"/>')
    p.append('<line x1="728" y1="100" x2="760" y2="100" class="svg-stroke" stroke-width="2.6" marker-end="url(#kgwebflow)"/>')

    # Titles & Captions
    for cx, title, sub in (
        (170, "1. User queries the site", "Search phrase parsed into searchable terms"),
        (542, "2. TF-IDF balances the terms", "Ubiquitous words dampened, specific intent amplified"),
        (920, "3. Highest match ranked #1", "Cosine similarity puts the answer first in results"),
    ):
        p.append(f'<text x="{cx}" y="206" text-anchor="middle" font-size="15" font-weight="700" class="svg-ink">{_esc(title)}</text>')
        p.append(f'<text x="{cx}" y="228" text-anchor="middle" font-size="12" font-weight="600" class="svg-text-muted">{_esc(sub)}</text>')

    p.append("</svg>")
    return "".join(p)


def _web_search_pipeline_svg() -> str:
    """Draws the 4 stages of web search retrieval as an illustrated flow strip."""
    stages = [
        {"n": "1", "name": "Crawl & Tokenize", "where": "Web pages to word tokens"},
        {"n": "2", "name": "Term Frequency (TF)", "where": "Count word occurrences in page"},
        {"n": "3", "name": "IDF Discrimination", "where": "Down-weight common site jargon"},
        {"n": "4", "name": "Cosine Scoring & SERP", "where": "Rank best solution at #1"},
    ]

    parts = [
        _svg_open(1160, 186, "The four stages of web and site search retrieval"),
        '<defs><marker id="kgpipetip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>',
    ]

    card_w, card_h = 248, 134
    for i, s in enumerate(stages):
        x = i * 290 + 10
        cx = x + card_w / 2

        parts.append(f'<rect x="{x}" y="20" width="{card_w}" height="{card_h}" rx="12" fill="currentColor" '
                     f'fill-opacity="0.04" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.4"/>')

        parts.append(f'<circle cx="{x + 24}" cy="38" r="12" fill="{ACCENT}"/>')
        parts.append(f'<text x="{x + 24}" y="43" text-anchor="middle" font-size="13" font-weight="700" '
                     f'fill="currentColor">{s["n"]}</text>')

        if i == 0:  # Crawl & Tokenize
            parts.append(f'<rect x="{x + 48}" y="48" width="46" height="56" rx="4" fill="currentColor" '
                         f'fill-opacity="0.08" stroke="currentColor" stroke-opacity="0.45"/>')
            parts.append(f'<rect x="{x + 48}" y="48" width="46" height="12" rx="4" fill="currentColor" '
                         f'fill-opacity="0.14"/>')
            for ly in (68, 76, 84, 92):
                parts.append(f'<line x1="{x + 55}" y1="{ly}" x2="{x + 87}" y2="{ly}" '
                             f'stroke="currentColor" stroke-opacity="0.5" stroke-width="1.8"/>')
            parts.append(f'<line x1="{x + 102}" y1="76" x2="{x + 118}" y2="76" stroke="currentColor" '
                         f'stroke-opacity="0.75" stroke-width="2" marker-end="url(#kgpipetip)"/>')
            tokens = [("reset", ACCENT, 0.4), ("password", ACCENT, 0.25), ("account", "currentColor", 0.18)]
            for tidx, (tok, clr, op) in enumerate(tokens):
                ty = 52 + tidx * 18
                parts.append(f'<rect x="{x + 124}" y="{ty}" width="68" height="14" rx="3" '
                             f'fill="{clr}" fill-opacity="{op}"/>')
                parts.append(f'<text x="{x + 158}" y="{ty + 11}" text-anchor="middle" font-size="9.5" '
                             f'font-weight="700" fill="currentColor">{tok}</text>')

        elif i == 1:  # Term Frequency (TF)
            parts.append(f'<rect x="{x + 42}" y="48" width="164" height="56" rx="5" fill="currentColor" '
                         f'fill-opacity="0.06" stroke="currentColor" stroke-opacity="0.35"/>')
            parts.append(f'<rect x="{x + 48}" y="54" width="34" height="13" rx="2" fill="{ACCENT}" fill-opacity="0.35"/>')
            parts.append(f'<text x="{x + 65}" y="64" text-anchor="middle" font-size="8.5" font-weight="700" fill="{ACCENT}">Reset</text>')
            parts.append(f'<text x="{x + 88}" y="64" font-size="8.5" font-weight="500" fill="currentColor">your</text>')
            parts.append(f'<rect x="{x + 110}" y="54" width="46" height="13" rx="2" fill="{ACCENT}" fill-opacity="0.35"/>')
            parts.append(f'<text x="{x + 133}" y="64" text-anchor="middle" font-size="8.5" font-weight="700" fill="{ACCENT}">password</text>')

            parts.append(f'<text x="{x + 50}" y="80" font-size="8.5" font-weight="500" fill="currentColor">from the</text>')
            parts.append(f'<rect x="{x + 85}" y="70" width="40" height="13" rx="2" fill="currentColor" fill-opacity="0.22"/>')
            parts.append(f'<text x="{x + 105}" y="80" text-anchor="middle" font-size="8.5" font-weight="600" fill="currentColor">account</text>')
            parts.append(f'<text x="{x + 130}" y="80" font-size="8.5" font-weight="500" fill="currentColor">settings...</text>')

            parts.append(f'<rect x="{x + 48}" y="88" width="152" height="12" rx="2" fill="currentColor" fill-opacity="0.10"/>')
            parts.append(f'<text x="{x + 124}" y="97" text-anchor="middle" font-size="8" font-weight="700" '
                         f'fill="{ACCENT}">TF in D1: reset=1 · password=2 · account=1</text>')

        elif i == 2:  # IDF Discrimination
            parts.append(f'<polygon points="{x + 124},100 {x + 132},100 {x + 128},88" fill="currentColor" fill-opacity="0.6"/>')
            parts.append(f'<line x1="{x + 116}" y1="100" x2="{x + 140}" y2="100" stroke="currentColor" stroke-opacity="0.6" stroke-width="2"/>')
            parts.append(f'<line x1="{x + 64}" y1="94" x2="{x + 192}" y2="68" stroke="currentColor" stroke-opacity="0.8" stroke-width="2.5"/>')
            parts.append(f'<line x1="{x + 64}" y1="94" x2="{x + 64}" y2="101" stroke="currentColor" stroke-opacity="0.6"/>')
            parts.append(f'<rect x="{x + 42}" y="101" width="46" height="14" rx="2" fill="currentColor" fill-opacity="0.20"/>')
            parts.append(f'<text x="{x + 65}" y="112" text-anchor="middle" font-size="8" font-weight="600" fill="currentColor">account (0.10)</text>')
            parts.append(f'<line x1="{x + 192}" y1="68" x2="{x + 192}" y2="75" stroke="currentColor" stroke-opacity="0.6"/>')
            parts.append(f'<rect x="{x + 168}" y="75" width="48" height="15" rx="2" fill="{ACCENT}" fill-opacity="0.9"/>')
            parts.append(f'<text x="{x + 192}" y="86" text-anchor="middle" font-size="8.5" font-weight="700" fill="currentColor">reset (0.40)</text>')

        else:  # Cosine Matching & SERP
            ox, oy = x + 56, 102
            parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + 50}" y2="{oy}" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.8"/>')
            parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - 46}" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.8"/>')
            parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + 38}" y2="{oy - 38}" stroke="{ACCENT}" stroke-width="2.4" '
                         f'marker-end="url(#kgpipetip)"/>')
            parts.append(f'<text x="{ox + 42}" y="{oy - 38}" font-size="9" font-weight="800" fill="{ACCENT}">Q</text>')
            parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + 46}" y2="{oy - 30}" stroke="currentColor" stroke-opacity="0.85" stroke-width="2" '
                         f'marker-end="url(#kgpipetip)"/>')
            parts.append(f'<text x="{ox + 50}" y="{oy - 27}" font-size="8.5" font-weight="700" fill="currentColor">D1</text>')
            parts.append(f'<path d="M {ox + 18} {oy - 12} A 22 22 0 0 1 {ox + 20} {oy - 17}" fill="none" stroke="currentColor" '
                         f'stroke-opacity="0.75" stroke-width="1.5"/>')
            parts.append(f'<text x="{ox + 24}" y="{oy - 12}" font-size="8.5" font-weight="700" fill="currentColor">θ</text>')

            bx = x + 120
            parts.append(f'<rect x="{bx}" y="52" width="76" height="12" rx="2" fill="{ACCENT}"/>')
            parts.append(f'<text x="{bx + 4}" y="61" font-size="8" font-weight="800" fill="currentColor">#1 D1 (0.32)</text>')
            parts.append(f'<rect x="{bx}" y="68" width="58" height="10" rx="2" fill="currentColor" fill-opacity="0.3"/>')
            parts.append(f'<text x="{bx + 4}" y="76" font-size="7.5" font-weight="700" fill="currentColor">#2 D4 (0.24)</text>')
            parts.append(f'<rect x="{bx}" y="82" width="30" height="9" rx="2" fill="currentColor" fill-opacity="0.2"/>')
            parts.append(f'<text x="{bx + 4}" y="89" font-size="7" font-weight="600" fill="currentColor">#3 D3 (0.05)</text>')

        parts.append(f'<text x="{cx}" y="132" text-anchor="middle" font-size="13" font-weight="700" '
                     f'fill="currentColor">{_esc(s["name"])}</text>')
        parts.append(f'<text x="{cx}" y="148" text-anchor="middle" font-size="10.5" font-weight="500" fill="currentColor" '
                     f'fill-opacity="0.85">{_esc(s["where"])}</text>')

        if i < len(stages) - 1:
            arrow_x = x + card_w + 6
            parts.append(f'<line x1="{arrow_x}" y1="84" x2="{arrow_x + 22}" y2="84" stroke="currentColor" '
                         f'stroke-opacity="0.6" stroke-width="2.2" marker-end="url(#kgpipetip)"/>')

    parts.append("</svg>")
    return "".join(parts)