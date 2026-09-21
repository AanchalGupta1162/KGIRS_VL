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
"""

import re
import math
import random
from collections import Counter
from html import escape as _html_escape
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Experiment 4: TF-IDF Based Document Retrieval",
    "course": "Knowledge Graphs and Information Retrieval Systems",
    "roll_no": "16-20",
    "group_no": "4",
    "aim": "To represent a document collection and a search query as **TF-IDF weighted vectors** in the "
           "vector space model, rank the documents by their **cosine similarity** to the query, observe how "
           "the choice of TF and IDF weighting scheme changes that ranking.",
    "objectives": [
        "Understand how Term Frequency (TF) and Inverse Document Frequency (IDF) capture local and "
        "global term importance respectively.",
        "Construct TF-IDF weighted vector representations for a document collection and a search query.",
        "Rank documents by similarity between the query vector and each document vector in the vector "
        "space model.",
        "Analyze how different TF and IDF weighting schemes influence retrieval ranking and result quality."
    ]
}

THEORY_CONTENT = {
    "procedure": [
        "Step 1: Read the Aim and the Theory page to review TF-IDF weighting and the vector space model.",
        "Step 2: Open the Simulation page from the sidebar menu.",
        "Step 3: Inspect (or edit) the sample document corpus - one document per line.",
        "Step 4: Enter a search query that is relevant to the corpus, or pick one of the sample queries.",
        "Step 5: Choose a Term Frequency scheme and an Inverse Document Frequency scheme.",
        "Step 6: Observe the ranked documents, the similarity bar chart, and the underlying TF, IDF and "
        "TF-IDF matrices.",
        "Step 7: Click 'Record this trial' after each configuration to log it into your session table.",
        "Step 8: Repeat for at least 3-4 distinct queries / weighting-scheme combinations.",
        "Step 9: Complete the assessment Quiz to test your conceptual understanding.",
        "Step 10: Open the Report page, enter your student information, and download your PDF report."
    ],
    "key_terms": {
        "Term Frequency (TF)": "How often a term occurs within a single document; may be a raw count, "
                                "length-normalized, or log-scaled.",
        "Inverse Document Frequency (IDF)": "How rare a term is across the whole collection; down-weights "
                                             "terms that appear in many documents.",
        "TF-IDF Weight": "Product of TF and IDF for a (term, document) pair - the term's importance in "
                          "that document relative to the corpus.",
        "Vector Space Model": "Representation of documents and queries as vectors over a shared "
                               "term vocabulary.",
        "Cosine Similarity": "Similarity between two vectors based on the cosine of the angle between "
                              "them; commonly used to rank documents by relevance.",
        "Document Frequency (df)": "The number of documents in the corpus that contain a given term "
                                    "at least once.",
        "Vocabulary": "The set of unique terms extracted from the corpus after tokenization and "
                       "stop-word removal.",
        "Out-of-Vocabulary (OOV) Term": "A query term that never occurs in the corpus; it cannot be "
                                         "assigned an IDF weight and is ignored during scoring.",
        "Relevance Judgment": "A human decision on whether a document satisfies the information need "
                               "behind a query; the ground truth for evaluation.",
        "Precision@k": "Share of the top k retrieved documents that are relevant.",
        "Recall@k": "Share of all relevant documents that appear in the top k results.",
        "Average Precision (AP)": "Mean of the precision values at the ranks of each relevant document; "
                                   "its mean over queries is MAP.",
        "nDCG@k": "Normalized Discounted Cumulative Gain; rewards relevant documents ranked near the top, "
                   "scaled so a perfect ranking scores 1."
    },
}

# Reading list for the experiment, grouped by what each source is for. Every journal reference below was
# checked against Crossref; "doi" holds the registered DOI, which the References page turns into a link.
REFERENCES = [
    ("Where TF-IDF came from", ":material/history_edu:", [
        {"cite": "H. P. Luhn, \"A Statistical Approach to Mechanized Encoding and Searching of Literary "
                 "Information\", *IBM Journal of Research and Development*, 1(4), 309-317, 1957.",
         "note": "Term frequency: the idea that how often a word occurs says something about the document.",
         "doi": "10.1147/rd.14.0309"},
        {"cite": "K. Sparck Jones, \"A statistical interpretation of term specificity and its application "
                 "in retrieval\", *Journal of Documentation*, 28(1), 11-21, 1972.",
         "note": "The paper that introduced inverse document frequency.",
         "doi": "10.1108/eb026526"},
        {"cite": "G. Salton, A. Wong and C. S. Yang, \"A vector space model for automatic indexing\", "
                 "*Communications of the ACM*, 18(11), 613-620, 1975.",
         "note": "Documents and queries as vectors - the model this lab simulates.",
         "doi": "10.1145/361219.361220"},
        {"cite": "G. Salton and C. Buckley, \"Term-weighting approaches in automatic text retrieval\", "
                 "*Information Processing & Management*, 24(5), 513-523, 1988.",
         "note": "Compares the TF and IDF weighting variants offered on the Simulation page.",
         "doi": "10.1016/0306-4573(88)90021-0"},
        {"cite": "S. Robertson, \"Understanding inverse document frequency: on theoretical arguments for "
                 "IDF\", *Journal of Documentation*, 60(5), 503-520, 2004.",
         "note": "Why IDF works, argued from first principles.",
         "doi": "10.1108/00220410410560582"},
    ]),
    ("How retrieval is evaluated", ":material/straighten:", [
        {"cite": "K. Jarvelin and J. Kekalainen, \"Cumulated gain-based evaluation of IR techniques\", "
                 "*ACM Transactions on Information Systems*, 20(4), 422-446, 2002.",
         "note": "The source of nDCG, one of the six parameters this lab reports.",
         "doi": "10.1145/582415.582418"},
        {"cite": "M. Sanderson, \"Test Collection Based Evaluation of Information Retrieval Systems\", "
                 "*Foundations and Trends in Information Retrieval*, 4(4), 247-375, 2010.",
         "note": "Where relevance judgments come from and how far you can trust them.",
         "doi": "10.1561/1500000009"},
    ]),
    ("Beyond TF-IDF", ":material/trending_up:", [
        {"cite": "S. Robertson and H. Zaragoza, \"The Probabilistic Relevance Framework: BM25 and "
                 "Beyond\", *Foundations and Trends in Information Retrieval*, 4(1-2), 1-174, 2009.",
         "note": "BM25, the ranking function that largely replaced plain TF-IDF.",
         "doi": "10.1561/1500000019"},
    ]),
    ("Textbooks", ":material/menu_book:", [
        {"cite": "G. Salton and M. J. McGill, *Introduction to Modern Information Retrieval*, "
                 "McGraw-Hill, 1983.",
         "note": "The classic text on the vector space model."},
        {"cite": "C. D. Manning, P. Raghavan and H. Schutze, *Introduction to Information Retrieval*, "
                 "Cambridge University Press, 2008.",
         "note": "Chapter 6 covers term weighting and the vector space model. Free online.",
         "url": "https://nlp.stanford.edu/IR-book/"},
        {"cite": "W. B. Croft, D. Metzler and T. Strohman, *Search Engines: Information Retrieval in "
                 "Practice*, Addison-Wesley, 2009.",
         "note": "The engineering view: how these ideas are built into real search systems."},
        {"cite": "R. Baeza-Yates and B. Ribeiro-Neto, *Modern Information Retrieval: The Concepts and "
                 "Technology behind Search*, 2nd ed., Addison-Wesley, 2011.",
         "note": "Broad reference covering models, evaluation and web search."},
    ]),
    ("Course material", ":material/school:", [
        {"cite": "IIT Kharagpur Virtual Labs - Information Retrieval discipline.",
         "note": "The Virtual Labs programme this experiment follows."},
    ]),
]

# The six stages of run_retrieval(), paired with the place in the Simulation page where each one is
# visible. Rendered as the flow strip at the top of the Real-world applications page.
PIPELINE_STAGES = [
    {"n": 1, "name": "Corpus",
     "what": "The collection you can retrieve from, one document per line.",
     "where": "Edit document corpus"},
    {"n": 2, "name": "Tokenize",
     "what": "Lower-case, strip punctuation, drop stop words and one-letter tokens.",
     "where": "Summary strip - vocabulary"},
    {"n": 3, "name": "TF and IDF",
     "what": "How often a term occurs in this document; how rare it is across all of them.",
     "where": "Matrices tab - TF, IDF"},
    {"n": 4, "name": "TF-IDF vectors",
     "what": "Multiply the two, so frequent-here-but-rare-elsewhere terms dominate.",
     "where": "Matrices tab - TF-IDF"},
    {"n": 5, "name": "Cosine scoring",
     "what": "Compare the query vector with every document vector, ignoring length.",
     "where": "Ranked results, Score chart"},
    {"n": 6, "name": "Rank and record",
     "what": "Sort by score, inspect the ranked list, and log the configuration as a trial.",
     "where": "Ranked results, Trial log"},
]

# Procedure steps grouped into phases: (label, icon, start index, end index) over THEORY_CONTENT["procedure"].
PROCEDURE_PHASES = [
    ("Prepare", ":material/menu_book:", 0, 2),
    ("Run the retrieval", ":material/play_arrow:", 2, 6),
    ("Record and assess", ":material/task_alt:", 6, None),
]

# Real-world uses of the same pipeline. Each carries a small corpus and query so the Applications page can
# run the real retrieval engine on it, rather than describing the result in prose.
APPLICATIONS = [
    {
        "name": "Web and site search",
        "icon": ":material/travel_explore:",
        "doc": "A web or help-centre page",
        "query_is": "What the user types",
        "metric": "nDCG@k",
        "why": "Every page says *account*, so it weighs nothing. Only one says *reset* - and that decides it.",
        "query": "reset my account password",
        "corpus": [
            "Reset your password from the account settings page by choosing Forgot password and confirming "
            "the link we email to you.",
            "Update the billing address and the payment card stored on your account from the same settings page.",
            "Our password policy asks for twelve characters, one number and one symbol on every new account.",
            "Contact the support desk if you still cannot sign in to your account after a password reset.",
            "The company was founded in 2011 and now employs four hundred people across six offices.",
        ],
    },
    {
        "name": "Spam and phishing filtering",
        "icon": ":material/report:",
        "doc": "One email or SMS",
        "query_is": "A profile of spam terms",
        "metric": "Precision@k",
        "why": "Shared vocabulary cancels out. What survives - *urgent*, *suspended*, *verify* - is the signal.",
        "query": "urgent verify your suspended account click this link",
        "corpus": [
            "URGENT your account has been suspended, click this link now to verify your identity before it "
            "is closed permanently.",
            "Verify your bank account immediately or the pending transfer will be cancelled, use the secure "
            "link below.",
            "Team lunch moves to Friday at one o'clock in the fourth floor kitchen, no need to reply.",
            "Your monthly invoice is attached as a PDF, no action is needed from your side.",
            "Congratulations you have won a prize, claim it urgently by clicking the link and paying the "
            "small handling fee.",
        ],
    },
    {
        "name": "More-like-this recommendation",
        "icon": ":material/recommend:",
        "doc": "Another article in the catalogue",
        "query_is": "The article being read",
        "metric": "Precision@k",
        "why": "Cosine divides length out, so a short closely-related piece still beats a long vague one.",
        "query": "knowledge graph entity linking for search",
        "corpus": [
            "Entity linking maps a mention in text to the matching node in a knowledge graph before the "
            "search runs.",
            "A knowledge graph stores entities as nodes and the relationships between them as labelled edges.",
            "Convolutional networks classify images by learning filters over neighbourhoods of pixels.",
            "Query understanding rewrites a search query using the entities recognised inside the text.",
            "The quarterly revenue report shows growth in three of the five regional markets.",
        ],
    },
    {
        "name": "Plagiarism and duplicate detection",
        "icon": ":material/content_copy:",
        "doc": "An earlier submission",
        "query_is": "The passage being checked",
        "metric": "Recall",
        "why": "Two texts on a topic share common words. Sharing the *rare* ones is what looks like copying.",
        "query": "documents and queries are represented as vectors over a shared vocabulary",
        "corpus": [
            "In the vector space model documents and queries are represented as vectors over a shared "
            "vocabulary of terms.",
            "Cosine similarity normalises for length so that long documents are not unfairly favoured.",
            "Boolean retrieval returns the set of documents matching a logical expression, without ranking them.",
            "A good sourdough loaf needs a mature starter and a long cold proof in the refrigerator.",
            "The vector representation of a document is sparse, because most vocabulary terms never occur in it.",
        ],
    },
    {
        "name": "Resume and job matching",
        "icon": ":material/badge:",
        "doc": "One candidate resume",
        "query_is": "The job description",
        "metric": "Recall@k",
        "why": "*Experience* and *team* are on every resume and add nothing. The skill terms carry the score.",
        "query": "python machine learning pipelines and model deployment experience",
        "corpus": [
            "Built and shipped machine learning pipelines in Python, owning model deployment behind a REST API.",
            "Five years of Python backend work on billing services, with occasional exposure to model deployment.",
            "Graphic designer specialising in brand identity, packaging and print production.",
            "Data analyst using SQL and spreadsheets to report on weekly marketing performance.",
            "Machine learning researcher publishing on graph neural networks, working mostly in Python.",
        ],
    },
    {
        "name": "Support ticket routing",
        "icon": ":material/support_agent:",
        "doc": "A support queue",
        "query_is": "An incoming ticket",
        "metric": "Reciprocal Rank",
        "why": "The corpus is one document per queue, so a single term unique to a queue routes the ticket.",
        "query": "wrong tax rate on my invoice",
        "corpus": [
            "Billing queue: invoice and tax rate problems, refunds, failed payments and subscription changes.",
            "Authentication queue: sign-in problems, password resets, two-factor codes and locked accounts.",
            "Performance queue: slow page loads, request timeouts and report generation failures.",
            "Onboarding queue: account setup, data import and user provisioning for new customers.",
            "Hardware queue: shipping, returns and physical replacement of faulty devices.",
        ],
    },
]

# The Theory page, written as six topics that each carry one picture. Photographs are hosted on Wikimedia
# Commons and fetched by the reader's browser, not by the server; every URL was checked to return an
# image and every credit line names the licence the file is published under. The one topic with no
# suitable photograph on Commons - documents as vectors - is drawn instead by _cosine_svg().
_WM = "https://upload.wikimedia.org/wikipedia/commons/thumb"
THEORY_TOPICS = [
    {
        "title": "The problem: a set is not an answer",
        "icon": ":material/inventory_2:",
        "image": f"{_WM}/a/a3/SanDiegoCityCollegeLearningResource_-_bookshelf.jpg"
                 f"/960px-SanDiegoCityCollegeLearningResource_-_bookshelf.jpg",
        "credit": "Joe Crawford, CC BY 2.0",
        "body": "Every retrieval system starts from the same awkward fact: the collection is larger than "
                "anyone is going to read.\n\n"
                "A **Boolean** search - documents containing *tax* **and** *invoice* - answers a yes/no "
                "question, and on a large collection it answers yes hundreds of times. It hands back a "
                "*set*, not an order, and a set of four hundred documents is barely more useful than the "
                "shelf you started with.\n\n"
                "**Ranked retrieval** replaces the yes/no question with a *how much* question: give every "
                "document a score for how well it answers this query, then sort by that score. The score "
                "has to be computed from the text itself, because nobody has labelled the collection in "
                "advance. TF-IDF is the classic answer to where that score comes from, and it remains the "
                "baseline that newer methods are measured against.",
    },
    {
        "title": "Why weighting is needed at all",
        "icon": ":material/history_edu:",
        "image": f"{_WM}/a/a6/Yale_card_catalog.jpg/960px-Yale_card_catalog.jpg",
        "credit": "Ragesoss, public domain",
        "body": "A card catalogue is an **index**: it records that a term occurs in a document, and where "
                "to find that document. That is enough to *find*, but not enough to *rank*.\n\n"
                "To an index, two documents containing the word *retrieval* look identical - even when one "
                "is a paper about retrieval and the other mentions it once in a footnote. The index has no "
                "way to express that the word means more in one than the other.\n\n"
                "**Term weighting** supplies that missing judgement. It gives every (term, document) pair "
                "a number saying how strongly that term characterises that document. Every ranked "
                "retrieval system is, underneath, a choice of weighting scheme. TF-IDF is the scheme that "
                "combines two intuitions pulling in opposite directions - one about this document, one "
                "about the whole collection.",
    },
    {
        "title": "From text to terms, and counting them",
        "icon": ":material/abc:",
        "image": f"{_WM}/a/ae/Metal_movable_type.jpg/960px-Metal_movable_type.jpg",
        "credit": "Willi Heidelbach, CC BY 2.5",
        "body": "Nothing can be weighed until the text is cut into units. This lab **tokenises** by "
                "lower-casing, splitting on punctuation, dropping single letters, and removing **stop "
                "words** - the *the*, *of*, *and* class that occurs everywhere and separates nothing.\n\n"
                "What survives is the **vocabulary**: the distinct terms in the corpus. It is built from "
                "the documents only, never from the query - which is why a query word that appears in no "
                "document has no weight to give and is simply dropped.\n\n"
                "**Term frequency** is the first intuition: a term repeated inside a document probably "
                "matters to it. The raw count is the simplest form, but it has two weaknesses. A long "
                "document scores higher on everything purely for being long, which **length "
                "normalisation** fixes by dividing by the document's token count. And relevance does not "
                "grow linearly - a document using *retrieval* twenty times is not twenty times more about "
                "retrieval than one using it once - which **log scaling** fixes by compressing large "
                "counts. The Simulation offers all three, so you can watch the ranking respond.",
        "latex": r"\mathrm{tf}_{t,d} = \text{count}(t,d) \quad\big|\quad "
                 r"\frac{\text{count}(t,d)}{\lvert d \rvert} \quad\big|\quad "
                 r"1 + \log_{10}\text{count}(t,d)",
    },
    {
        "title": "Inverse document frequency: rarity is information",
        "icon": ":material/filter_alt:",
        "image": f"{_WM}/a/af/Karen_Sp%C3%A4rck.jpg/500px-Karen_Sp%C3%A4rck.jpg",
        "credit": "Markus Kuhn, CC BY 2.5",
        "body": "Term frequency on its own is easily fooled, because the most repeated word is rarely the "
                "most informative one. In a collection of medical papers *patient* appears in every "
                "document; counting it tells you nothing about which paper to open.\n\n"
                "In 1972 **Karen Sparck Jones** formalised the fix: a term's weight should fall as it "
                "spreads across the collection. With *N* documents in total and *df(t)* of them "
                "containing the term, IDF is the log of that ratio.\n\n"
                "The behaviour is worth committing to memory. Take *N* = 100. A term in **1** document "
                "scores log&#8321;&#8320;(100) = **2.0**. In **10** documents, log&#8321;&#8320;(10) = "
                "**1.0**. In **50**, about **0.3**. In **all 100**, log&#8321;&#8320;(1) = **0 exactly** - "
                "a term present everywhere adds nothing to any score, however often it is repeated.\n\n"
                "The logarithm is doing real work: without it a term in one document would outweigh "
                "everything else by a factor of a hundred. The log makes the penalty gradual. The "
                "**smoothed** variant, log(1 + N/df), keeps every weight positive and is the safer choice "
                "on a small collection.",
        "latex": r"\mathrm{idf}_t = \log_{10}\frac{N}{\mathrm{df}_t}",
    },
    {
        "title": "TF-IDF and the vector space model",
        "icon": ":material/scatter_plot:",
        "image": "svg:cosine",
        "credit": None,
        "body": "Multiply the two intuitions together and you have the weight of term *t* in document "
                "*d*. It is large only when **both** parts are large: the term is frequent *here* and "
                "rare *elsewhere*. That is exactly the definition of a term that tells this document "
                "apart from the rest of the collection.\n\n"
                "Do this for every term and every document, and each document becomes a **vector** of "
                "weights - one component per vocabulary term. The query is turned into a vector the same "
                "way, using the corpus IDF values, so that query and documents live in the same space. "
                "Similarity now becomes a geometric question.\n\n"
                "A plain **dot product** would work, but it quietly favours long documents: more terms "
                "means more non-zero components means a bigger product. **Cosine similarity** divides the "
                "dot product by the length of both vectors, cancelling magnitude entirely and leaving "
                "only the angle between them. Two documents pointing the same way score 1.0 whatever "
                "their length; two sharing no terms score 0. Switch cosine normalisation off in the "
                "Simulation and you can watch the length bias come straight back.",
        "latex": r"w_{t,d} = \mathrm{tf}_{t,d} \times \mathrm{idf}_t \qquad\qquad "
                 r"\cos(\vec q, \vec d) = \frac{\vec q \cdot \vec d}"
                 r"{\lVert \vec q \rVert \, \lVert \vec d \rVert}",
    },
    {
        "title": "Judging the ranking",
        "icon": ":material/straighten:",
        "image": f"{_WM}/2/26/Precisionrecall.svg/960px-Precisionrecall.svg.png",
        "credit": "Walber, CC BY-SA 4.0",
        "image_width": 190,
        "body": "A ranking cannot be judged on its own. It is judged against **relevance judgments** - a "
                "person's decision about which documents genuinely answer the query. Given those and a "
                "cut-off *k*:\n\n"
                "- **Precision@k** - of the *k* documents returned, what share is relevant? Punishes "
                "returning junk.\n"
                "- **Recall@k** - of all the relevant documents, what share reached the top *k*? "
                "Punishes missing things.\n"
                "- **F1@k** - the harmonic mean of the two, for when both matter.\n\n"
                "Those three ignore the *order* within the top *k*. Three more do not:\n\n"
                "- **Average Precision** averages the precision measured at each rank where a relevant "
                "document appears, so promoting a relevant document raises the score.\n"
                "- **nDCG@k** discounts each relevant document by the logarithm of its rank and divides "
                "by the best ordering possible, so a perfect ranking scores exactly 1.\n"
                "- **Reciprocal Rank** is 1 / the rank of the first relevant document - the right measure "
                "when the user needs only one good answer.\n\n"
                "Precision and recall trade against each other: returning more documents can only help "
                "recall, and usually costs precision. Which side you favour depends on the job, which is "
                "why the Simulation reports all six at once.",
    },
]


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

    # The angle between q and the better-matching document.
    p.append(f'<path d="M {ox + 88} {oy - 46} A 100 100 0 0 1 {ox + 96} {oy - 29}" fill="none" '
             f'stroke="{ACCENT}" stroke-width="2"/>')
    p.append(f'<text x="{ox + 108} " y="{oy - 30}" font-size="14" fill="{ACCENT}">&#952;</text>')
    p.append(f'<text x="{ox + 4}" y="{oy + 22}" font-size="11" fill="currentColor" fill-opacity="0.55">'
             f'origin</text>')
    p.append("</svg>")
    return "".join(p)


# Small, generic English stop-word list used during tokenization.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "that", "this", "these",
    "those", "it", "its", "their", "they", "he", "she", "we", "you", "i", "your", "our",
    "his", "her", "them", "which", "who", "whom", "what", "when", "where", "why", "how",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must", "do",
    "does", "did", "have", "has", "had", "not", "no", "if", "than", "then", "so", "such",
    "there", "here", "about", "into", "over", "under", "also", "using", "used", "use"
}

# Default sample corpus: 41 documents, one per line, across five course topics (information retrieval,
# knowledge graphs, machine learning / NLP, databases, semantic search / RAG) plus two long documents.
SAMPLE_CORPUS_PATH = Path(__file__).with_name("sample_corpus.txt")
DEFAULT_CORPUS = [line.strip() for line in SAMPLE_CORPUS_PATH.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
DEFAULT_QUERY = "TF-IDF document ranking"

# Sample queries that work well against the built-in corpus, offered as quick picks in the sidebar.
# The document lists are kept as a record of which documents are on-topic for each query.
SAMPLE_JUDGMENTS = {
    "TF-IDF document ranking": ["D3", "D4", "D40"],
    "evaluation measures for ranked retrieval": ["D5", "D37", "D41"],
    "graph database query language": ["D10", "D27"],
    "semantic search with embeddings": ["D33", "D34", "D36", "D38"],
    "reducing hallucinations in language models": ["D35", "D39"],
    "vocabulary mismatch synonyms": ["D6", "D17", "D33", "D36"],
}

TF_SCHEMES = ["Raw Term Frequency", "Normalized Term Frequency", "Log-Normalized Term Frequency"]
IDF_SCHEMES = ["Standard IDF: log(N / df)", "Smoothed IDF: log(1 + N / df)"]

# How many of the 50 questions a student is asked in one sitting.
QUIZ_LENGTH = 10

# The question bank. Every sitting draws QUIZ_LENGTH of these at random, so repeating the quiz is useful
# practice rather than a memory test. Option order is fixed because the options are labelled A) to D).
QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What does the acronym TF-IDF stand for?",
        "options": [
            "A) Term Frequency - Inverse Document Frequency",
            "B) Text Format - Index Data Field",
            "C) Total Frequency - Indexed Data File",
            "D) Term Filter - Inverted Document Format",
        ],
        "answer_index": 0,
        "explanation": "TF-IDF combines Term Frequency (local importance) with Inverse Document Frequency "
                       "(global rarity) into a single weight.",
    },
    {
        "id": 2,
        "question": "What is the main purpose of the IDF component in TF-IDF weighting?",
        "options": [
            "A) To count how many times a term occurs in a single document",
            "B) To down-weight terms that occur in many documents and up-weight rare, discriminative terms",
            "C) To remove punctuation and stop words from the text",
            "D) To increase the length of every document artificially",
        ],
        "answer_index": 1,
        "explanation": "IDF reduces the weight of common terms that appear across most documents and "
                       "boosts the weight of rarer, more informative terms.",
    },
    {
        "id": 3,
        "question": "A term occurs in every single document of the corpus, so df = N. Using the standard "
                    "formula, what is its IDF?",
        "options": [
            "A) 1, because it is present everywhere",
            "B) N, because it scales with the collection",
            "C) 0, so the term contributes nothing to any score",
            "D) Undefined, because the formula divides by zero",
        ],
        "answer_index": 2,
        "explanation": "idf = log10(N / df) = log10(N / N) = log10(1) = 0. A term present in every "
                       "document cannot distinguish between them, so it adds nothing to any score.",
    },
    {
        "id": 4,
        "question": "For non-negative TF-IDF vectors, what is the range of the cosine similarity?",
        "options": [
            "A) -1 to 1",
            "B) 0 to 1",
            "C) 0 to the number of documents",
            "D) 1 to infinity",
        ],
        "answer_index": 1,
        "explanation": "Cosine ranges from -1 to 1 in general, but TF-IDF weights are never negative, so "
                       "no vector can point 'against' another. The result lies between 0 (no shared "
                       "terms) and 1 (identical direction).",
    },
    {
        "id": 5,
        "question": "What bias does cosine normalisation remove from the ranking?",
        "options": [
            "A) The bias towards documents written more recently",
            "B) The bias towards documents containing rare terms",
            "C) The bias towards documents at the start of the corpus",
            "D) The bias towards long documents, which accumulate larger vectors",
        ],
        "answer_index": 3,
        "explanation": "Dividing by the length of both vectors cancels magnitude and leaves only the "
                       "angle, so a long document gains no advantage from simply containing more terms.",
    },
    {
        "id": 6,
        "question": "In this lab, the vocabulary is built from which text?",
        "options": [
            "A) The document corpus only",
            "B) The query only",
            "C) The corpus and the query combined",
            "D) A fixed English dictionary shipped with the app",
        ],
        "answer_index": 0,
        "explanation": "The vocabulary comes from the corpus alone. That is why a query word that appears "
                       "in no document has no IDF, and is reported as out-of-vocabulary.",
    },
    {
        "id": 7,
        "question": "A user types a query word that appears in no document in the corpus. What happens?",
        "options": [
            "A) Every document score is set to zero",
            "B) The term is ignored during scoring and reported as out-of-vocabulary",
            "C) The term is added to the vocabulary with an IDF of 1",
            "D) The search is rejected with an error",
        ],
        "answer_index": 1,
        "explanation": "An out-of-vocabulary term has no document frequency and therefore no IDF weight. "
                       "It cannot contribute to any score, so it is dropped and listed separately.",
    },
    {
        "id": 8,
        "question": "Why are stop words removed before weighting?",
        "options": [
            "A) They are always spelled incorrectly",
            "B) They make the corpus file larger on disk",
            "C) They occur in nearly every document, so they separate nothing and only add noise",
            "D) They cannot be converted to lower case",
        ],
        "answer_index": 2,
        "explanation": "Words like 'the', 'of' and 'and' appear almost everywhere. Their IDF would be "
                       "near zero anyway, so removing them early shrinks the vocabulary at no cost.",
    },
    {
        "id": 9,
        "question": "What is the main weakness of using the raw term count as TF?",
        "options": [
            "A) It cannot be computed for short documents",
            "B) It always produces negative weights",
            "C) It ignores how rare the term is in the collection",
            "D) It favours long documents, because they contain more of every term",
        ],
        "answer_index": 3,
        "explanation": "A raw count grows with document length, so long documents score higher on "
                       "everything. Length normalisation divides by the document's token count to fix it.",
    },
    {
        "id": 10,
        "question": "What problem does log-normalised term frequency address?",
        "options": [
            "A) That relevance does not grow linearly with repetition",
            "B) That some documents contain no terms at all",
            "C) That the corpus may contain duplicate documents",
            "D) That IDF can be negative on small collections",
        ],
        "answer_index": 0,
        "explanation": "A document using a term twenty times is not twenty times more relevant than one "
                       "using it once. The logarithm compresses large counts so extra repetitions matter "
                       "progressively less.",
    },
    {
        "id": 11,
        "question": "What does the document frequency df(t) of a term count?",
        "options": [
            "A) The total number of times t occurs across the whole corpus",
            "B) The number of documents that contain t at least once",
            "C) The number of times t occurs in the longest document",
            "D) The position of t in the vocabulary",
        ],
        "answer_index": 1,
        "explanation": "df counts documents, not occurrences. A term appearing fifty times in one "
                       "document still has df = 1.",
    },
    {
        "id": 12,
        "question": "Which expression is the standard IDF used in this lab?",
        "options": [
            "A) log10(df / N)",
            "B) N / df",
            "C) log10(N / df)",
            "D) 1 - (df / N)",
        ],
        "answer_index": 2,
        "explanation": "Standard IDF is log10(N / df): the ratio of collection size to document "
                       "frequency, compressed by a logarithm.",
    },
    {
        "id": 13,
        "question": "Why might you prefer the smoothed IDF, log(1 + N/df), on a very small corpus?",
        "options": [
            "A) It runs faster than the standard formula",
            "B) It makes every term equally important",
            "C) It removes the need for stop-word removal",
            "D) It keeps every weight strictly positive, so no term collapses to exactly zero",
        ],
        "answer_index": 3,
        "explanation": "With standard IDF a term in all N documents scores exactly 0 and drops out "
                       "entirely. Smoothing adds 1 inside the logarithm so the weight stays positive.",
    },
    {
        "id": 14,
        "question": "How is the TF-IDF weight of term t in document d computed?",
        "options": [
            "A) By adding tf and idf together",
            "B) By multiplying tf by idf",
            "C) By dividing tf by idf",
            "D) By taking the larger of tf and idf",
        ],
        "answer_index": 1,
        "explanation": "w(t,d) = tf(t,d) x idf(t). Multiplying means the weight is high only when the "
                       "term is both frequent here and rare elsewhere.",
    },
    {
        "id": 15,
        "question": "N = 100 documents and a term appears in exactly 10 of them. What is its standard IDF?",
        "options": [
            "A) 0.1",
            "B) 1.0",
            "C) 10.0",
            "D) 2.0",
        ],
        "answer_index": 1,
        "explanation": "idf = log10(100 / 10) = log10(10) = 1.0.",
    },
    {
        "id": 16,
        "question": "N = 100 documents and a term appears in exactly 1 of them. What is its standard IDF?",
        "options": [
            "A) 2.0",
            "B) 1.0",
            "C) 0.01",
            "D) 100.0",
        ],
        "answer_index": 0,
        "explanation": "idf = log10(100 / 1) = log10(100) = 2.0, the highest weight any term can reach in "
                       "a 100-document collection.",
    },
    {
        "id": 17,
        "question": "How does Precision@k differ from Recall@k?",
        "options": [
            "A) Precision is measured before ranking, recall afterwards",
            "B) Precision counts documents, recall counts terms",
            "C) Precision is the share of the top k that is relevant; recall is the share of all relevant "
            "documents found in the top k",
            "D) They are two names for the same quantity",
        ],
        "answer_index": 2,
        "explanation": "Precision divides by k, the number returned. Recall divides by |R|, the total "
                       "number of relevant documents. Precision punishes junk; recall punishes omissions.",
    },
    {
        "id": 18,
        "question": "F1@k is defined as which combination of precision and recall?",
        "options": [
            "A) Their arithmetic mean",
            "B) Their harmonic mean, 2PR / (P + R)",
            "C) Their product, P x R",
            "D) Their difference, P - R",
        ],
        "answer_index": 1,
        "explanation": "F1 is the harmonic mean. Unlike the arithmetic mean it stays low unless both "
                       "precision and recall are reasonably high.",
    },
    {
        "id": 19,
        "question": "Which evaluation parameter is sensitive to where in the ranking the relevant "
                    "documents appear?",
        "options": [
            "A) Precision@k",
            "B) Recall@k",
            "C) F1@k",
            "D) nDCG@k",
        ],
        "answer_index": 3,
        "explanation": "Precision, recall and F1 treat every document inside the top k equally. nDCG "
                       "discounts each relevant document by the log of its rank, so promoting one raises "
                       "the score.",
    },
    {
        "id": 20,
        "question": "What does Reciprocal Rank measure?",
        "options": [
            "A) 1 divided by the rank of the first relevant document",
            "B) The number of relevant documents divided by k",
            "C) The rank of the last relevant document",
            "D) The average of all precision values",
        ],
        "answer_index": 0,
        "explanation": "RR = 1 / rank of the first relevant result. It is the right measure when the user "
                       "only needs one good answer. Averaged over queries it becomes MRR.",
    },
    {
        "id": 21,
        "question": "In the DCG formula, a relevant document at rank i contributes rel(i) divided by what?",
        "options": [
            "A) i squared",
            "B) log2(i + 1)",
            "C) the square root of i",
            "D) the total number of documents",
        ],
        "answer_index": 1,
        "explanation": "DCG@k = sum of rel(i) / log2(i + 1). The logarithmic discount means a document "
                       "slipping from rank 1 to rank 2 costs more than slipping from rank 9 to rank 10.",
    },
    {
        "id": 22,
        "question": "Why is DCG divided by the ideal DCG to produce nDCG?",
        "options": [
            "A) To convert the score into a percentage of the corpus size",
            "B) To remove the effect of stop words",
            "C) To scale the score so a perfect ranking gives exactly 1, making queries comparable",
            "D) To make the score independent of the weighting scheme",
        ],
        "answer_index": 2,
        "explanation": "The ideal DCG is what a perfect ordering would score. Dividing by it normalises "
                       "the result to the 0-1 range so different queries can be compared.",
    },
    {
        "id": 23,
        "question": "Two documents have identical wording, but one repeats the whole text twice. Under "
                    "cosine similarity, how do their scores for the same query compare?",
        "options": [
            "A) The longer one scores roughly twice as high",
            "B) They score the same, because cosine ignores magnitude",
            "C) The longer one scores zero",
            "D) The shorter one always scores higher",
        ],
        "answer_index": 1,
        "explanation": "Doubling a document scales its vector but does not change its direction. Cosine "
                       "depends only on the angle, so the score is unchanged.",
    },
    {
        "id": 24,
        "question": "When the query is turned into a vector, which IDF values are used?",
        "options": [
            "A) IDF values computed from the query text itself",
            "B) An IDF of 1 for every query term",
            "C) The IDF values computed from the document corpus",
            "D) IDF values supplied by the user",
        ],
        "answer_index": 2,
        "explanation": "The query is projected into the same space as the documents using the "
                       "corpus-derived IDF values. Otherwise the two vectors would not be comparable.",
    },
    {
        "id": 25,
        "question": "In the vector space model, what does each dimension of a document vector correspond to?",
        "options": [
            "A) One document in the corpus",
            "B) One term in the shared vocabulary",
            "C) One character of the text",
            "D) One evaluation parameter",
        ],
        "answer_index": 1,
        "explanation": "Each vector has one component per vocabulary term, so the number of dimensions "
                       "equals the vocabulary size.",
    },
    {
        "id": 26,
        "question": "Why are TF-IDF document vectors usually sparse?",
        "options": [
            "A) Because most vocabulary terms do not occur in any given document",
            "B) Because IDF values are usually zero",
            "C) Because stop words are removed",
            "D) Because the corpus is stored one document per line",
        ],
        "answer_index": 0,
        "explanation": "A single document uses a tiny fraction of the whole vocabulary, so almost every "
                       "component of its vector is zero.",
    },
    {
        "id": 27,
        "question": "A document shares no terms at all with the query. What is its cosine score?",
        "options": [
            "A) 1",
            "B) 0",
            "C) -1",
            "D) It depends on the document's length",
        ],
        "answer_index": 1,
        "explanation": "With no shared terms the dot product is zero, so the cosine is zero. In this lab "
                       "such documents are collapsed at the bottom of the results.",
    },
    {
        "id": 28,
        "question": "What does it mean for a document to count as 'retrieved' in this lab's evaluation?",
        "options": [
            "A) It appears anywhere in the corpus",
            "B) It was judged relevant by a person",
            "C) Its similarity score is greater than zero",
            "D) It contains every query term",
        ],
        "answer_index": 2,
        "explanation": "A document is retrieved once it scores above zero, meaning it shares at least one "
                       "term with the query.",
    },
    {
        "id": 29,
        "question": "What are relevance judgments?",
        "options": [
            "A) The similarity scores produced by the retrieval engine",
            "B) A human decision about which documents actually satisfy the information need",
            "C) The IDF values of the query terms",
            "D) The order in which documents were added to the corpus",
        ],
        "answer_index": 1,
        "explanation": "Relevance judgments are the ground truth, decided by a person and independent of "
                       "how the system happens to rank. Without them nothing can be evaluated.",
    },
    {
        "id": 30,
        "question": "A system returns more documents for the same query. What typically happens to "
                    "precision and recall?",
        "options": [
            "A) Both rise",
            "B) Both fall",
            "C) Recall can only rise or stay level, while precision usually falls",
            "D) Precision rises and recall falls",
        ],
        "answer_index": 2,
        "explanation": "Returning more can only find more relevant documents, so recall cannot fall. The "
                       "extra results are usually less relevant, so precision typically drops.",
    },
    {
        "id": 31,
        "question": "How does Average Precision differ from Precision@k?",
        "options": [
            "A) It averages precision at each rank where a relevant document appears, so order matters",
            "B) It uses recall instead of precision",
            "C) It only considers the single top-ranked document",
            "D) It ignores relevance judgments entirely",
        ],
        "answer_index": 0,
        "explanation": "AP averages the precision measured at every rank holding a relevant document. "
                       "Moving a relevant document up the ranking raises AP; Precision@k would not "
                       "change.",
    },
    {
        "id": 32,
        "question": "What is the relationship between AP and MAP?",
        "options": [
            "A) MAP is AP measured at a single cut-off",
            "B) MAP is AP averaged over a set of queries",
            "C) MAP is AP with stop words removed",
            "D) They are unrelated measures",
        ],
        "answer_index": 1,
        "explanation": "AP describes one query. Mean Average Precision is the mean of AP across many "
                       "queries, which is how whole systems are compared.",
    },
    {
        "id": 33,
        "question": "What does Boolean retrieval return that ranked retrieval does not?",
        "options": [
            "A) A score for every document",
            "B) An unordered set of documents matching a logical expression",
            "C) A precision-recall curve",
            "D) A normalised vector for each document",
        ],
        "answer_index": 1,
        "explanation": "Boolean retrieval answers a yes/no question and hands back a set with no order. "
                       "Ranked retrieval scores every document so the best can be shown first.",
    },
    {
        "id": 34,
        "question": "Which of these is NOT one of the term frequency schemes offered in this lab?",
        "options": [
            "A) Raw count",
            "B) Length-normalised count",
            "C) Log-normalised count",
            "D) Square-root normalised count",
        ],
        "answer_index": 3,
        "explanation": "The Simulation offers raw, normalised and log-normalised term frequency. "
                       "Square-root scaling exists in the literature but is not one of the three here.",
    },
    {
        "id": 35,
        "question": "Which of these terms would be the most discriminative in a 200-document collection?",
        "options": [
            "A) A term with df = 200",
            "B) A term with df = 150",
            "C) A term with df = 3",
            "D) A term with df = 100",
        ],
        "answer_index": 2,
        "explanation": "IDF falls as df rises, so the rarest term is the most discriminative. df = 3 "
                       "gives log10(200/3) which is about 1.82; df = 200 gives exactly 0.",
    },
    {
        "id": 36,
        "question": "What happens to a term's IDF when new documents containing that term are added to "
                    "the corpus?",
        "options": [
            "A) It falls, because df grows relative to N",
            "B) It rises, because the corpus is larger",
            "C) It stays fixed once computed",
            "D) It becomes negative",
        ],
        "answer_index": 0,
        "explanation": "Adding documents that contain the term raises df faster than it raises N, so the "
                       "ratio N/df shrinks and the IDF falls. The term has become less distinctive.",
    },
    {
        "id": 37,
        "question": "If cosine normalisation is switched off, what is being used to score documents instead?",
        "options": [
            "A) The Euclidean distance between vectors",
            "B) The raw dot product of the query and document vectors",
            "C) The document frequency alone",
            "D) The number of matching terms",
        ],
        "answer_index": 1,
        "explanation": "Without normalisation the score is the plain dot product, which reintroduces the "
                       "bias towards long documents.",
    },
    {
        "id": 38,
        "question": "Which researcher introduced the idea behind inverse document frequency, in 1972?",
        "options": [
            "A) Gerard Salton",
            "B) Hans Peter Luhn",
            "C) Karen Sparck Jones",
            "D) Stephen Robertson",
        ],
        "answer_index": 2,
        "explanation": "Karen Sparck Jones proposed term specificity in 1972. Luhn had earlier worked on "
                       "term frequency, and Salton developed the vector space model.",
    },
    {
        "id": 39,
        "question": "Which ranking function is widely regarded as the practical successor to plain TF-IDF?",
        "options": [
            "A) BM25",
            "B) PageRank",
            "C) K-means",
            "D) Naive Bayes",
        ],
        "answer_index": 0,
        "explanation": "BM25, from the probabilistic relevance framework, adds saturation and document "
                       "length normalisation and is the standard baseline in modern search systems.",
    },
    {
        "id": 40,
        "question": "Two documents tie with exactly the same similarity score. How does this lab order them?",
        "options": [
            "A) Alphabetically by their text",
            "B) By length, shortest first",
            "C) Randomly on each run",
            "D) They keep their original order in the corpus",
        ],
        "answer_index": 3,
        "explanation": "The sort is stable, so documents with equal scores appear in the order they were "
                       "listed in the corpus.",
    },
    {
        "id": 41,
        "question": "N = 100 and a term appears in 50 documents. Its standard IDF is closest to which value?",
        "options": [
            "A) 0.50",
            "B) 0.30",
            "C) 2.00",
            "D) 0.00",
        ],
        "answer_index": 1,
        "explanation": "idf = log10(100 / 50) = log10(2), which is about 0.301. A term in half the "
                       "collection carries only a small weight.",
    },
    {
        "id": 42,
        "question": "Why is the vocabulary built before any TF or IDF value is computed?",
        "options": [
            "A) Because every vector needs the same, shared set of components",
            "B) Because stop words can only be removed afterwards",
            "C) Because cosine similarity requires sorted terms",
            "D) Because the query must be read first",
        ],
        "answer_index": 0,
        "explanation": "Vectors can only be compared if they are indexed by the same terms in the same "
                       "order, so the shared vocabulary has to be fixed first.",
    },
    {
        "id": 43,
        "question": "What does tokenisation do in this lab's pipeline?",
        "options": [
            "A) Ranks the documents by length",
            "B) Lower-cases text, strips punctuation and drops stop words and single letters",
            "C) Computes the document frequency of each term",
            "D) Converts each document into a PDF",
        ],
        "answer_index": 1,
        "explanation": "Tokenisation turns raw text into the comparable units that everything downstream "
                       "counts and weights.",
    },
    {
        "id": 44,
        "question": "A term has a high TF in one document but also a high df across the corpus. What is "
                    "its TF-IDF weight likely to be?",
        "options": [
            "A) High, because TF is high",
            "B) Low, because the high df drives IDF towards zero",
            "C) Exactly 1",
            "D) Negative",
        ],
        "answer_index": 1,
        "explanation": "The two factors multiply. A near-zero IDF pulls the product down no matter how "
                       "large the term frequency is.",
    },
    {
        "id": 45,
        "question": "What is the effect of stop-word removal on the size of the vocabulary?",
        "options": [
            "A) It grows",
            "B) It is unchanged",
            "C) It shrinks",
            "D) It becomes equal to the number of documents",
        ],
        "answer_index": 2,
        "explanation": "Stop words are removed before the vocabulary is built, so the set of distinct "
                       "terms is smaller and the vectors have fewer dimensions.",
    },
    {
        "id": 46,
        "question": "Why can a plain dot product favour a long document over a short one?",
        "options": [
            "A) Long documents have higher IDF values",
            "B) Long documents are always more relevant",
            "C) Long documents have more non-zero components, so the sum of products is larger",
            "D) The dot product divides by document length",
        ],
        "answer_index": 2,
        "explanation": "Nothing in the dot product cancels magnitude, so more terms simply means a larger "
                       "total. Cosine fixes this by dividing by both vector lengths.",
    },
    {
        "id": 47,
        "question": "Precision@k is measured with k = 5 and 3 of the top 5 documents are relevant. What "
                    "is Precision@5?",
        "options": [
            "A) 0.6",
            "B) 0.3",
            "C) 1.67",
            "D) 5.0",
        ],
        "answer_index": 0,
        "explanation": "Precision@k = relevant in top k / k = 3 / 5 = 0.6.",
    },
    {
        "id": 48,
        "question": "There are 8 relevant documents in total and 2 of them appear in the top 5. What is "
                    "Recall@5?",
        "options": [
            "A) 0.4",
            "B) 0.25",
            "C) 0.8",
            "D) 2.0",
        ],
        "answer_index": 1,
        "explanation": "Recall@k = relevant in top k / total relevant = 2 / 8 = 0.25.",
    },
    {
        "id": 49,
        "question": "Which statement about the vector space model is correct?",
        "options": [
            "A) It requires documents to be labelled with a category first",
            "B) It can only rank documents that contain every query term",
            "C) It represents documents and queries as vectors over one shared term space",
            "D) It stores documents as nodes and edges in a graph",
        ],
        "answer_index": 2,
        "explanation": "Salton's vector space model puts documents and queries in the same term space so "
                       "similarity becomes a geometric comparison.",
    },
    {
        "id": 50,
        "question": "Why is TF-IDF still taught and used even though newer ranking methods exist?",
        "options": [
            "A) It is the only method that can handle text",
            "B) It always outperforms every newer method",
            "C) It requires no computation at all",
            "D) It is simple, interpretable, needs no training data, and is the baseline newer methods "
            "are measured against",
        ],
        "answer_index": 3,
        "explanation": "Every weight can be traced back to a count and a document frequency, it needs no "
                       "labelled training data, and it remains the reference point for newer methods.",
    },
]


# ======================================================================================
# 2. TF-IDF RETRIEVAL ENGINE (CORE SIMULATION LOGIC)
# ======================================================================================

def tokenize(text: str) -> list:
    """Lower-cases, strips punctuation, and removes stop words / single-character tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def build_vocabulary(doc_tokens_list: list) -> list:
    """Vocabulary is derived strictly from the corpus (not the query)."""
    vocab = set()
    for toks in doc_tokens_list:
        vocab.update(toks)
    return sorted(vocab)


def compute_tf(tokens: list, vocab: list, scheme: str) -> dict:
    counts = Counter(tokens)
    n = len(tokens)
    tf_vec = {}
    for term in vocab:
        c = counts.get(term, 0)
        if scheme == "Raw Term Frequency":
            val = float(c)
        elif scheme == "Normalized Term Frequency":
            val = (c / n) if n > 0 else 0.0
        else:  # Log-Normalized Term Frequency
            val = (1.0 + math.log10(c)) if c > 0 else 0.0
        tf_vec[term] = val
    return tf_vec


def compute_df(doc_tokens_list: list, vocab: list) -> dict:
    df = {}
    for term in vocab:
        df[term] = sum(1 for toks in doc_tokens_list if term in toks)
    return df


def compute_idf(df: dict, vocab: list, n_docs: int, scheme: str) -> dict:
    idf = {}
    for term in vocab:
        d = df.get(term, 0)
        if d <= 0:
            idf[term] = 0.0
            continue
        if scheme.startswith("Standard"):
            idf[term] = math.log10(n_docs / d)
        else:
            idf[term] = math.log10(1.0 + (n_docs / d))
    return idf


def tfidf_vector(tf_vec: dict, idf_vec: dict, vocab: list) -> dict:
    return {t: tf_vec[t] * idf_vec[t] for t in vocab}


def cosine_similarity(v1: dict, v2: dict, vocab: list) -> float:
    dot = sum(v1[t] * v2[t] for t in vocab)
    n1 = math.sqrt(sum(v1[t] ** 2 for t in vocab))
    n2 = math.sqrt(sum(v2[t] ** 2 for t in vocab))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def run_retrieval(corpus: list, query: str, tf_scheme: str, idf_scheme: str, use_cosine: bool = True) -> dict:
    """
    Runs the full TF-IDF retrieval pipeline: tokenize -> TF -> DF -> IDF -> TF-IDF -> score -> rank.
    Replaces the generic 'run_simulation' hook from the base template.
    """
    doc_tokens_list = [tokenize(d) for d in corpus]
    vocab = build_vocabulary(doc_tokens_list)

    query_tokens_all = tokenize(query)
    oov_terms = sorted(set(t for t in query_tokens_all if t not in vocab))
    query_tokens = [t for t in query_tokens_all if t in vocab]

    n_docs = len(corpus)
    df = compute_df(doc_tokens_list, vocab)
    idf = compute_idf(df, vocab, n_docs, idf_scheme)

    tf_matrix = [compute_tf(toks, vocab, tf_scheme) for toks in doc_tokens_list]
    tfidf_matrix = [tfidf_vector(tf, idf, vocab) for tf in tf_matrix]

    query_tf = compute_tf(query_tokens, vocab, tf_scheme)
    query_tfidf = tfidf_vector(query_tf, idf, vocab)

    scores = []
    for doc_vec in tfidf_matrix:
        if use_cosine:
            score = cosine_similarity(query_tfidf, doc_vec, vocab)
        else:
            score = sum(query_tfidf[t] * doc_vec[t] for t in vocab)
        scores.append(score)

    ranking = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)

    return {
        "vocab": vocab, "df": df, "idf": idf,
        "tf_matrix": tf_matrix, "tfidf_matrix": tfidf_matrix,
        "query_tfidf": query_tfidf, "scores": scores,
        "ranking": ranking, "oov_terms": oov_terms,
        "query_tokens": query_tokens
    }


# ======================================================================================
# 3. LAB REPORT PDF EXPORTER
# ======================================================================================

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

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 5, "-", 0)
        pdf.multi_cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials & Data", new_x="LMARGIN", new_y="NEXT")

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

    # 3. Discussion & Notes
    if pdf.get_y() > pdf.h - 45:  # keep the heading together with its text
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
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


# ======================================================================================
# 4. SECTION RENDERERS: AIM, THEORY, PROCEDURE, APPLICATIONS, REFERENCES, SIMULATION, QUIZ, REPORT
# ======================================================================================

TF_SHORT = {
    "Raw Term Frequency": "Raw count",
    "Normalized Term Frequency": "Normalized",
    "Log-Normalized Term Frequency": "Log-normalized",
}
IDF_SHORT = {
    "Standard IDF: log(N / df)": "Standard",
    "Smoothed IDF: log(1 + N / df)": "Smoothed",
}
DEFAULT_NOTES = ("The TF-IDF retrieval trials demonstrated consistent, query-relevant document ranking across "
                 "the tested weighting-scheme configurations.")

# Characters that Streamlit markdown would otherwise interpret inside document text.
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


def render_aim_section():
    """Renders the Aim page: the aim statement, the objectives and what the student hands in."""
    st.header("Aim", icon=":material/flag:")
    st.caption("What this experiment sets out to do, and why it is worth doing.")

    with st.container(border=True):
        st.markdown(EXPERIMENT_CONFIG["aim"])

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Objectives", icon=":material/checklist_rtl:")
        for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"], start=1):
            st.markdown(f"**{i}.** &nbsp; {obj}")
    with right:
        st.subheader("What you hand in", icon=":material/inventory_2:")
        st.markdown(
            "- At least **3 recorded trials** across different queries and weighting schemes.\n"
            "- The **six evaluation parameters** for each trial, measured against relevance judgments.\n"
            "- A completed **quiz** score.\n"
            "- A downloaded **PDF lab report** with your trial table and observations."
        )

    st.divider()
    render_applications_block()


def render_theory_section():
    """Renders the Theory page: six topics, each with a picture, then the glossary."""
    st.header("Theory", icon=":material/menu_book:")
    st.caption("How TF-IDF turns text into vectors and ranks documents against a query, in six steps. "
               "Read it in order - each step uses the one before it.")

    for i, topic in enumerate(THEORY_TOPICS):
        with st.container(border=True):
            st.subheader(f"{i + 1}. {topic['title']}", icon=topic["icon"])
            # A narrow picture column keeps the illustration to a thumbnail and leaves the page to the text.
            picture, words = st.columns([1, 4], vertical_alignment="top")
            with picture:
                if topic["image"] == "svg:cosine":
                    _draw(_cosine_svg())
                else:
                    st.image(topic["image"], width=topic.get("image_width", "stretch"))
                if topic["credit"]:
                    st.caption(f":gray-badge[:material/photo_camera: {topic['credit']}]")
            with words:
                st.markdown(topic["body"])
                if topic.get("latex"):
                    st.latex(topic["latex"])

    with st.expander(f"Key terms ({len(THEORY_CONTENT['key_terms'])})", icon=":material/dictionary:"):
        var_df = pd.DataFrame(list(THEORY_CONTENT["key_terms"].items()), columns=["Term", "Definition"])
        st.dataframe(
            var_df, hide_index=True, width="stretch",
            column_config={"Term": st.column_config.TextColumn(width="medium"),
                           "Definition": st.column_config.TextColumn(width="large")}
        )


def render_procedure_section():
    """Renders the Procedure page: the ordered steps, grouped into three phases."""
    st.header("Procedure", icon=":material/checklist:")
    st.caption("Follow the steps in order. Steps 3 to 9 all happen on the Simulation page.")

    steps = THEORY_CONTENT["procedure"]
    for label, icon, start, end in PROCEDURE_PHASES:
        st.subheader(label, icon=icon)
        with st.container(border=True):
            for step in steps[start:end]:
                name, _, text = step.partition(": ")
                st.markdown(f"**{name}** &nbsp; {text}")


ACCENT = "#2A9AA4"

# Inline styling for the illustrated blocks on the Real-world applications page. Colors are either the
# shared accent or a neutral gray alpha, so the same markup reads correctly in the light and dark themes.
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
           justify-content: center; font-weight: 700; font-size: .85rem;
           background: rgba(128, 128, 128, 0.18); }
.kg-rank.kg-top { background: #2A9AA4; color: #fff; }
.kg-body { flex: 1 1 auto; min-width: 0; }
.kg-bar { height: 8px; border-radius: 4px; background: rgba(128, 128, 128, 0.16); overflow: hidden;
          margin-bottom: .3rem; }
.kg-bar > span { display: block; height: 100%; border-radius: 4px; background: #2A9AA4; }
.kg-bar.kg-zero > span { background: rgba(128, 128, 128, 0.35); }
.kg-text { font-size: .86rem; line-height: 1.45; opacity: .92; }
.kg-score { flex: 0 0 3.4rem; text-align: right; font-variant-numeric: tabular-nums;
            font-weight: 600; font-size: .85rem; }
.kg-hit { background: rgba(224, 146, 58, 0.30); color: inherit; padding: 0 .18em; border-radius: 3px; }
.kg-idfrow { display: flex; align-items: center; gap: .75rem; padding: .3rem 0; }
.kg-idflab { flex: 0 0 11rem; font-size: .82rem; opacity: .75; }
.kg-idfval { flex: 0 0 2.6rem; text-align: right; font-variant-numeric: tabular-nums;
             font-weight: 600; font-size: .85rem; }
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
    """Renders an SVG drawing.

    st.html sanitizes with DOMPurify's html-only profile, which strips <svg> outright, so the drawings go
    through st.image instead - Streamlit inlines them as a data URI. An <img> does not inherit the page's
    text color, so currentColor is resolved against the active theme here before handing it over.
    """
    mode = None
    try:
        mode = st.context.theme["type"]
    except Exception:  # theme is unavailable outside a live script run
        pass
    ink = {"light": "#15232A", "dark": "#E3ECEE"}.get(mode, "#7F8C93")
    st.image(svg.replace("currentColor", ink), width=width)


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


def render_applications_block():
    """Renders the orientation block at the foot of the Aim page: what the lab does, why it is worth
    using, a tour of its pages, and where the same technique is used in practice - plus a live retrieval
    over the chosen application's own collection. Not a page of its own; called by render_aim_section()."""
    st.header("Real-world applications", icon=":material/public:")
    st.caption("New here? Start with this.")
    st.html(APP_VISUAL_CSS)

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
    for row_start in (0, 3):
        cols = st.columns(3)
        for col, a in zip(cols, APPLICATIONS[row_start:row_start + 3]):
            with col:
                with st.container(border=True):
                    st.markdown(f"## {a['icon']}")
                    st.markdown(f"**{a['name']}**")
                    st.markdown(
                        f":gray-badge[:material/description: {a['doc']}]  \n"
                        f":gray-badge[:material/search: {a['query_is']}]  \n"
                        f":blue-badge[:material/straighten: {a['metric']}]"
                    )

    st.divider()

    # --- Live retrieval over the chosen application -------------------------------------------------
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
        st.caption("Bigger chip, rarer term, more say over the ranking. Faded chips are not in this "
                   "collection and are dropped.")

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


def render_references_section():
    """Renders the References page."""
    st.header("References", icon=":material/library_books:")
    st.caption("Where the theory, the formulas and the evaluation parameters used in this lab come from.")

    total = sum(len(items) for _, _, items in REFERENCES)
    st.markdown(f":gray-badge[:material/format_list_numbered: {total} sources]")

    n = 0
    for group, icon, items in REFERENCES:
        st.subheader(group, icon=icon)
        for item in items:
            n += 1
            with st.container(border=True):
                st.markdown(f"**{n}.** &nbsp; {item['cite']}")
                st.caption(item["note"])
                link = item.get("url") or (f"https://doi.org/{item['doi']}" if item.get("doi") else None)
                if link:
                    label = f"doi:{item['doi']}" if item.get("doi") else "Read online"
                    st.markdown(f"[:material/link: {label}]({link})")


def set_corpus_text(text: str, query: str | None = None):
    """Queues new corpus (and optionally query) text for the widgets.

    A keyed widget ignores a changed `value=` argument, so the browser would keep showing the old text.
    The text is stored here and written into the widget keys at the start of the next run, before the
    widgets are created.
    """
    st.session_state["corpus_text"] = text
    st.session_state["pending_corpus_text"] = text
    if query is not None:
        st.session_state["query_text"] = query
        st.session_state["pending_query_text"] = query


CUSTOM_QUERY_OPTION = "Type your own..."


def _use_sample_query():
    """Copies the selected sample query into the search box."""
    picked = st.session_state.get("sample_query_pick")
    if picked and picked != CUSTOM_QUERY_OPTION:
        st.session_state["query_text_input"] = picked
        st.session_state["query_text"] = picked



SIMULATION_VISUAL_CSS = """
<style>
/* --- Simulation-only visual polish: richer graph-palette colors, no layout/functionality changes --- */
div[class*="st-key-sim_hero_"] {
    background:
        radial-gradient(circle at 8% 18%, rgba(42,154,164,.24), transparent 34%),
        radial-gradient(circle at 92% 12%, rgba(224,146,58,.20), transparent 30%),
        radial-gradient(circle at 52% 100%, rgba(108,99,255,.12), transparent 32%),
        linear-gradient(135deg, rgba(42,154,164,.12), rgba(255,255,255,.02));
    border: 1px solid rgba(42,154,164,.38);
    border-radius: 18px;
    padding: 1rem 1.1rem .9rem;
    margin: .25rem 0 1rem;
    box-shadow: 0 10px 30px rgba(42,154,164,.10), 0 4px 18px rgba(108,99,255,.06);
    animation: tfidf-soft-glow 3.8s ease-in-out infinite alternate;
}
@keyframes tfidf-soft-glow {
    from { box-shadow: 0 8px 24px rgba(42,154,164,.08), 0 3px 14px rgba(91,143,249,.05); }
    to   { box-shadow: 0 12px 34px rgba(224,146,58,.14), 0 5px 20px rgba(108,99,255,.08); }
}
div[class*="st-key-sim_hero_"]::after {
    content: "";
    position: absolute;
    width: 110px;
    height: 110px;
    right: 7%;
    top: 2px;
    border-radius: 50%;
    background:
        radial-gradient(circle, rgba(91,192,190,.22), rgba(91,143,249,.10) 42%,
        rgba(224,146,58,.05) 62%, transparent 74%);
    pointer-events: none;
    animation: tfidf-orb-drift 5s ease-in-out infinite alternate;
}
@keyframes tfidf-orb-drift {
    from { transform: translate3d(0, 2px, 0) scale(.92); opacity: .65; }
    to   { transform: translate3d(-14px, 10px, 0) scale(1.08); opacity: 1; }
}

/* Pipeline pills use the same palette as the similarity graph. */
div[class*="st-key-sim_pipeline_"] .sim_step_pill {
    display: inline-block;
    position: relative;
    overflow: hidden;
    border-radius: 999px;
    border: 1px solid rgba(42,154,164,.30);
    background:
        linear-gradient(135deg, rgba(42,154,164,.15), rgba(91,143,249,.08) 48%, rgba(224,146,58,.10));
    padding: .30rem .68rem;
    font-size: .78rem;
    font-weight: 650;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}
div[class*="st-key-sim_pipeline_"] .sim_step_pill::after {
    content: "";
    position: absolute;
    left: 10%;
    right: 10%;
    bottom: 0;
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, #5B8FF9, #2A9AA4, #6C63FF, #E0923A, #D95F8A, #4CAF7D);
    opacity: .72;
}
div[class*="st-key-sim_pipeline_"] .sim_step_pill:hover {
    transform: translateY(-2px);
    border-color: rgba(224,146,58,.55);
    background:
        linear-gradient(135deg, rgba(42,154,164,.20), rgba(108,99,255,.12), rgba(224,146,58,.14));
    box-shadow: 0 6px 18px rgba(42,154,164,.14), 0 3px 12px rgba(108,99,255,.08);
}
div[class*="st-key-sim_pipeline_"] {
    display: flex;
    flex-wrap: wrap;
    gap: .45rem;
    margin: .2rem 0 .9rem;
}

/* Existing step elements: stronger color pops without changing their structure. */
div[class*="st-key-sim_step_"] {
    border-radius: 999px;
    border: 1px solid rgba(42,154,164,.30);
    background:
        linear-gradient(135deg, rgba(42,154,164,.14), rgba(91,143,249,.07) 45%, rgba(224,146,58,.09));
    padding: .28rem .62rem;
    font-size: .78rem;
    font-weight: 650;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div[class*="st-key-sim_step_"]:hover {
    transform: translateY(-2px);
    border-color: rgba(224,146,58,.52);
    box-shadow: 0 5px 16px rgba(42,154,164,.12), 0 3px 12px rgba(217,95,138,.07);
}

/* Main animation card: layered palette wash keeps teal dominant but adds the graph colors. */
div[class*="st-key-anim_step_card_"] {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 100% 0%, rgba(224,146,58,.13), transparent 27%),
        radial-gradient(circle at 0% 100%, rgba(91,143,249,.10), transparent 30%),
        radial-gradient(circle at 82% 82%, rgba(108,99,255,.08), transparent 25%),
        linear-gradient(145deg, rgba(42,154,164,.075), rgba(255,255,255,.018));
    border: 1px solid rgba(42,154,164,.34) !important;
    box-shadow:
        0 8px 26px rgba(42,154,164,.09),
        0 4px 18px rgba(91,143,249,.05),
        inset 0 1px 0 rgba(255,255,255,.08);
    border-radius: 16px !important;
    animation: tfidf-step-in .35s ease-out, tfidf-card-pulse 2.6s ease-in-out .35s 1;
}
div[class*="st-key-anim_step_card_"]::before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    height: 4px;
    border-radius: 16px 16px 0 0;
    background: linear-gradient(
        90deg,
        #5B8FF9 0%,
        #2A9AA4 20%,
        #6C63FF 40%,
        #5BC0BE 58%,
        #E0923A 74%,
        #D95F8A 88%,
        #4CAF7D 100%
    );
    background-size: 220% 100%;
    animation: tfidf-palette-flow 4.5s ease-in-out infinite;
    opacity: .92;
    pointer-events: none;
}
div[class*="st-key-anim_step_card_"]::after {
    content: "";
    position: absolute;
    width: 150px;
    height: 150px;
    right: -55px;
    bottom: -65px;
    border-radius: 50%;
    background: radial-gradient(circle,
        rgba(217,95,138,.11),
        rgba(108,99,255,.07) 42%,
        transparent 72%);
    pointer-events: none;
}
@keyframes tfidf-palette-flow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes tfidf-card-pulse {
    0% { box-shadow: 0 8px 26px rgba(42,154,164,.08), 0 3px 14px rgba(91,143,249,.04); }
    45% { box-shadow: 0 12px 32px rgba(224,146,58,.14), 0 5px 20px rgba(108,99,255,.08); }
    100% { box-shadow: 0 8px 26px rgba(42,154,164,.08), 0 3px 14px rgba(91,143,249,.04); }
}
div[class*="st-key-anim_step_card_"] h2,
div[class*="st-key-anim_step_card_"] h3 {
    background: linear-gradient(90deg, #2A9AA4 0%, #5B8FF9 28%, #6C63FF 52%, #E0923A 76%, #D95F8A 100%);
    background-size: 180% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: tfidf-heading-flow 5s ease-in-out infinite;
}
@keyframes tfidf-heading-flow {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Existing data areas get subtle palette framing only. */
div[class*="st-key-anim_step_card_"] .stDataFrame,
div[class*="st-key-anim_step_card_"] [data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(42,154,164,.18);
    box-shadow: inset 0 0 0 1px rgba(91,143,249,.035);
}
div[class*="st-key-anim_step_card_"] [data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #5B8FF9, #2A9AA4, #6C63FF, #5BC0BE, #E0923A, #D95F8A, #4CAF7D);
    background-size: 250% 100%;
    animation: tfidf-progress-flow 2.2s linear infinite;
}
@keyframes tfidf-progress-flow {
    0% { background-position: 0% 50%; }
    100% { background-position: 250% 50%; }
}
div[class*="st-key-anim_step_card_"] [data-testid="stMetric"] {
    border-radius: 12px;
    border: 1px solid rgba(42,154,164,.22);
    background:
        linear-gradient(135deg, rgba(42,154,164,.07), rgba(91,143,249,.045) 50%, rgba(224,146,58,.045));
}
div[class*="st-key-sim_controls_"] {
    border-radius: 14px;
    background:
        radial-gradient(circle at 0% 50%, rgba(42,154,164,.07), transparent 30%),
        radial-gradient(circle at 100% 50%, rgba(224,146,58,.06), transparent 30%);
}
div[class*="st-key-sim_controls_"] button {
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}
div[class*="st-key-sim_controls_"] button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow:
        0 7px 18px rgba(42,154,164,.18),
        0 3px 10px rgba(91,143,249,.08),
        0 2px 8px rgba(224,146,58,.10);
}
div[class*="st-key-anim_step_card_"] [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(224,146,58,.40);
    box-shadow: 0 7px 18px rgba(42,154,164,.10), 0 3px 12px rgba(217,95,138,.07);
}
div[class*="st-key-anim_step_card_"] [data-testid="stDataFrame"]:hover {
    box-shadow: 0 8px 22px rgba(42,154,164,.10), 0 3px 12px rgba(108,99,255,.06);
}
div[class*="st-key-anim_step_card_"] [data-testid="stMetric"],
div[class*="st-key-anim_step_card_"] [data-testid="stDataFrame"] {
    transition: transform .20s ease, box-shadow .20s ease, border-color .20s ease;
}

/* Existing tabs: a small multicolor wash, not a new UI element. */
div[class*="st-key-sim_controls_"] [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, rgba(42,154,164,.08), rgba(91,143,249,.06), rgba(108,99,255,.06), rgba(224,146,58,.08));
    border-radius: 12px;
    padding: 3px;
}
div[class*="st-key-sim_controls_"] [data-baseweb="tab"] {
    transition: background .18s ease, transform .18s ease;
}
div[class*="st-key-sim_controls_"] [data-baseweb="tab"]:hover {
    background: linear-gradient(90deg, rgba(91,143,249,.10), rgba(108,99,255,.09), rgba(224,146,58,.10));
    transform: translateY(-1px);
}

@media (prefers-reduced-motion: reduce) {
    div[class*="st-key-sim_hero_"],
    div[class*="st-key-anim_step_card_"],
    div[class*="st-key-anim_step_card_"] [data-testid="stProgress"] > div > div,
    div[class*="st-key-anim_step_card_"]::before,
    div[class*="st-key-anim_step_card_"] h2,
    div[class*="st-key-anim_step_card_"] h3 {
        animation: none !important;
    }
    div[class*="st-key-sim_step_"],
    div[class*="st-key-sim_controls_"] button,
    div[class*="st-key-anim_step_card_"] [data-testid="stMetric"],
    div[class*="st-key-anim_step_card_"] [data-testid="stDataFrame"] {
        transition: none !important;
    }
    div[class*="st-key-sim_hero_"]::after {
        animation: none !important;
    }
}
</style>
"""


ANIMATION_STEPS = [
    ("Tokenize", "Tokenize & build vocabulary", ":material/splitscreen:"),
    ("TF matrix", "Term frequency (TF) matrix", ":material/table_rows:"),
    ("DF & IDF", "Document frequency & IDF", ":material/query_stats:"),
    ("TF x IDF", "TF x IDF matrix", ":material/grid_on:"),
    ("Query vector", "Query vector", ":material/search:"),
    ("Scoring", "Similarity scoring", ":material/calculate:"),
    ("Ranking", "Rank documents", ":material/sort:"),
    ("Top-k", "Top-k results", ":material/emoji_events:"),
]
ANIMATION_INTERVAL = "1.4s"


def render_simulation_section():
    """Renders Section 2: a step-by-step walk-through of the TF-IDF pipeline for one query."""
    st.header("Simulation", icon=":material/manage_search:")
    st.caption("Set a query and a weighting scheme, then step through the pipeline to watch TF-IDF build "
               "the term-document matrix, weight it by IDF, score it against your query, and rank the results.")
    st.html(SIMULATION_VISUAL_CSS)
    with st.container(key="sim_hero_"):
        st.markdown("**Interactive retrieval workspace**")
        st.caption("Explore how tokenization, TF, IDF, TF-IDF weighting and cosine similarity work together.")
    with st.container(horizontal=True, gap="small", key="sim_pipeline_"):
        for _step_i, (_short, _long, _icon) in enumerate(ANIMATION_STEPS, start=1):
            st.markdown(f'<div class="sim_step_pill"><b>{_step_i}</b>&nbsp; {_short}</div>', unsafe_allow_html=True)

    # --- Controls ---------------------------------------------------------------------------
    if "pending_corpus_text" in st.session_state:
        st.session_state["corpus_text_area"] = st.session_state.pop("pending_corpus_text")
    if "pending_query_text" in st.session_state:
        st.session_state["query_text_input"] = st.session_state.pop("pending_query_text")
    st.session_state.setdefault("corpus_text_area", st.session_state["corpus_text"])
    st.session_state.setdefault("query_text_input", st.session_state["query_text"])

    corpus_preview = [line.strip() for line in st.session_state["corpus_text"].split("\n") if line.strip()]

    with st.container(border=True, key="sim_controls_"):
        is_default_corpus = st.session_state["corpus_text"].strip() == "\n".join(DEFAULT_CORPUS)
        query_col, sample_col = st.columns([3, 2], vertical_alignment="bottom")
        with query_col:
            query = st.text_input(
                "Search query", key="query_text_input",
                icon=":material/search:", placeholder="e.g. TF-IDF document ranking"
            )
        with sample_col:
            if is_default_corpus:
                st.selectbox(
                    "Sample queries", options=[CUSTOM_QUERY_OPTION] + list(SAMPLE_JUDGMENTS),
                    key="sample_query_pick", on_change=_use_sample_query,
                    help="Pick one of the sample queries the corpus was written around."
                )
        st.session_state["query_text"] = query

        with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
            tf_label = st.segmented_control(
                "Term frequency", options=TF_SCHEMES, format_func=TF_SHORT.get,
                default=TF_SCHEMES[0], key="tf_scheme", required=True
            )
            idf_label = st.segmented_control(
                "Inverse document frequency", options=IDF_SCHEMES, format_func=IDF_SHORT.get,
                default=IDF_SCHEMES[0], key="idf_scheme", required=True,
                help="Standard: log(N / df). Smoothed: log(1 + N / df)."
            )
            use_cosine = st.toggle("Cosine normalization", value=True, key="use_cosine",
                                   help="Off = raw dot product, which favors longer documents.")
            top_k = st.number_input(
                "Top-k results", min_value=1, max_value=max(1, len(corpus_preview)),
                value=min(5, max(1, len(corpus_preview))), step=1, key="top_k",
                help="How many top-ranked documents the animation reveals at the end."
            )
        tf_scheme, idf_scheme = tf_label, idf_label

        with st.expander("Edit document corpus", icon=":material/edit_note:",
                         expanded=st.session_state.pop("corpus_expanded", False)):
            corpus_text = st.text_area(
                "One document per line", height=200, key="corpus_text_area"
            )
            st.session_state["corpus_text"] = corpus_text
            if st.button("Restore sample corpus", icon=":material/restart_alt:"):
                set_corpus_text("\n".join(DEFAULT_CORPUS), DEFAULT_QUERY)
                st.session_state["corpus_expanded"] = True
                st.rerun()

    corpus = [line.strip() for line in st.session_state["corpus_text"].split("\n") if line.strip()]
    if len(corpus) < 2:
        st.warning("Add at least two documents to run retrieval: type them one per line.",
                   icon=":material/warning:")
        return
    if not query.strip():
        st.info("Type a search query to run the TF-IDF pipeline.", icon=":material/search:")
        return

    result = run_retrieval(corpus, query, tf_scheme, idf_scheme, use_cosine)
    doc_ids = [f"D{i + 1}" for i in range(len(corpus))]
    top_k = min(top_k, len(corpus))

    # Reset to the first step whenever the underlying pipeline configuration changes, so the
    # walk-through never shows a stale step for a different query/corpus/scheme combination.
    fingerprint = (tuple(corpus), query.strip(), tf_scheme, idf_scheme, use_cosine, top_k)
    if st.session_state["anim_fingerprint"] != fingerprint:
        st.session_state["anim_fingerprint"] = fingerprint
        st.session_state["anim_step"] = 0
        st.session_state["anim_playing"] = False

    ctx = {
        "corpus": corpus, "doc_ids": doc_ids, "query": query, "result": result,
        "tf_scheme": tf_scheme, "idf_scheme": idf_scheme, "use_cosine": use_cosine, "top_k": top_k,
    }

    step = st.session_state["anim_step"]
    n = len(ANIMATION_STEPS)

    _render_animation_controls(step, n)
    _render_animation_display(ctx, n)


def _inject_step_transition_css():
    st.html("""
        <style>
        @keyframes tfidf-step-in {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        div[class*="st-key-anim_step_card_"] {
            animation: tfidf-step-in 0.35s ease-out;
        }
        </style>
    """)


def _render_animation_controls(step: int, n: int):
    """Previous/Play-Pause/Next/Reset. Lives outside the fragment below so every click triggers
    a full rerun, which is required to change the fragment's `run_every` interval."""
    playing = st.session_state["anim_playing"]
    at_end = step >= n - 1

    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if st.button("Previous", icon=":material/chevron_left:", disabled=step <= 0):
            st.session_state["anim_step"] = step - 1
            st.session_state["anim_playing"] = False
            st.rerun()

        if playing:
            if st.button("Pause", icon=":material/pause:", type="primary"):
                st.session_state["anim_playing"] = False
                st.rerun()
        else:
            if st.button("Replay" if at_end else "Play",
                        icon=":material/replay:" if at_end else ":material/play_arrow:",
                        type="primary"):
                if at_end:
                    st.session_state["anim_step"] = 0
                st.session_state["anim_playing"] = True
                st.rerun()

        if st.button("Next", icon=":material/chevron_right:", disabled=step >= n - 1):
            st.session_state["anim_step"] = step + 1
            st.session_state["anim_playing"] = False
            st.rerun()

        if st.button("Reset", icon=":material/restart_alt:", disabled=step == 0 and not playing):
            st.session_state["anim_step"] = 0
            st.session_state["anim_playing"] = False
            st.rerun()


def _render_animation_display(ctx: dict, n: int):
    # run_every is read fresh on every full rerun (this whole entrypoint file re-executes each
    # time), so toggling "anim_playing" via the controls above changes the interval on the very
    # next run: None when paused, a fixed delay while playing.
    interval = ANIMATION_INTERVAL if st.session_state["anim_playing"] else None

    @st.fragment(run_every=interval)
    def _fragment():
        step = st.session_state["anim_step"]
        _, title, icon = ANIMATION_STEPS[step]

        st.progress((step + 1) / n, text=f"Step {step + 1} of {n}: **{title}**")

        _inject_step_transition_css()
        # Keying the card by step forces Streamlit to remount it (rather than patch its
        # children) whenever the step changes, which is what replays the CSS fade/slide.
        with st.container(border=True, key=f"anim_step_card_{step}"):
            st.subheader(title, icon=icon)
            _render_animation_step(step, ctx)

        if st.session_state["anim_playing"]:
            if step < n - 1:
                st.session_state["anim_step"] = step + 1
            else:
                # Reached the end: stop, and force a full rerun so the controls/fragment above
                # are redefined with run_every=None on their next render (a fragment-scoped
                # rerun can't change its own run_every).
                st.session_state["anim_playing"] = False
                st.rerun()

    _fragment()


def _render_animation_step(step: int, ctx: dict):
    corpus, doc_ids, query = ctx["corpus"], ctx["doc_ids"], ctx["query"]
    result, tf_scheme, idf_scheme = ctx["result"], ctx["tf_scheme"], ctx["idf_scheme"]
    use_cosine, top_k = ctx["use_cosine"], ctx["top_k"]
    vocab = result["vocab"]
    query_terms = set(result["query_tokens"])

    if step == 0:  # Tokenize & build vocabulary
        st.markdown("Every document (and the query) is lower-cased, stripped of punctuation, and filtered "
                    "through a stop-word list. The vocabulary is the set of unique terms left in the corpus.")
        tok_df = pd.DataFrame([
            {"Doc": doc_ids[i], "Tokens": ", ".join(tokenize(corpus[i])) or "_(none)_"}
            for i in range(len(corpus))
        ])
        st.dataframe(tok_df, hide_index=True, width="stretch")
        query_tokens_all = tokenize(query)
        if query_tokens_all:
            st.markdown("**Query tokens:** " + " ".join(
                f":green-badge[{t}]" if t in query_terms else f":gray-badge[{t} (out of vocabulary)]"
                for t in query_tokens_all
            ))
        st.caption(f"Vocabulary: **{len(vocab)}** unique terms across **{len(corpus)}** documents.")

    elif step == 1:  # TF matrix
        st.markdown(f"Term frequency ({TF_SHORT[tf_scheme].lower()}) counts how often each term occurs "
                    "in each document — one row per document, one column per vocabulary term.")
        tf_df = pd.DataFrame(result["tf_matrix"], index=doc_ids, columns=vocab).round(3)
        st.dataframe(tf_df, width="stretch")

    elif step == 2:  # DF & IDF
        st.markdown(f"Document frequency (df) counts how many documents contain each term. "
                    f"IDF ({IDF_SHORT[idf_scheme]}) then down-weights terms that appear in many documents.")
        idf_df = pd.DataFrame([{"Term": t, "df": result["df"][t], "IDF": result["idf"][t]} for t in vocab])
        idf_df = idf_df.sort_values("IDF", ascending=False)
        st.dataframe(
            idf_df, hide_index=True, width="stretch",
            column_config={"IDF": st.column_config.ProgressColumn(
                "IDF", format="%.4f", min_value=0.0, max_value=float(idf_df["IDF"].max() or 1.0))}
        )

    elif step == 3:  # TF-IDF matrix
        st.latex(r"w_{t,d} = \mathrm{tf}_{t,d} \times \mathrm{idf}_t")
        st.markdown("Multiplying the TF matrix by the IDF vector (broadcast across every document) gives "
                    "the TF-IDF weight matrix: high where a term is frequent here but rare elsewhere.")
        tfidf_df = pd.DataFrame(result["tfidf_matrix"], index=doc_ids, columns=vocab).round(4)
        st.dataframe(tfidf_df, width="stretch")

    elif step == 4:  # Query vector
        st.markdown("The query is tokenized the same way, then projected into the same TF-IDF space using "
                    "the corpus's IDF values.")
        if query_terms:
            query_tf = compute_tf(result["query_tokens"], vocab, tf_scheme)
            q_df = pd.DataFrame([
                {"Term": t, "TF": query_tf[t], "IDF": result["idf"][t], "Weight": result["query_tfidf"][t]}
                for t in sorted(query_terms)
            ])
            st.dataframe(q_df.round(4), hide_index=True, width="stretch")
        else:
            st.warning("None of the query terms appear in the corpus, so the query vector is all zeros.",
                      icon=":material/search_off:")

    elif step == 5:  # Similarity scoring
        score_label = "Cosine similarity" if use_cosine else "Dot product"
        st.markdown(f"Each document vector is compared against the query vector using **{score_label}**.")
        _chart_palette = ["#2A9AA4", "#E0923A", "#6C63FF", "#4CAF7D", "#D95F8A", "#5B8FF9"]
        _bar_colors = [
            _chart_palette[i % len(_chart_palette)] if score > 0 else "rgba(128,128,128,0.28)"
            for i, score in enumerate(result["scores"])
        ]
        fig = go.Figure(go.Bar(
            x=doc_ids, y=result["scores"],
            text=[f"{s:.3f}" for s in result["scores"]], textposition="outside",
            marker=dict(color=_bar_colors, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            height=340, yaxis_title=score_label, showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.16)", zeroline=False),
        )
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    elif step == 6:  # Ranking
        st.markdown("Documents are sorted in descending order of similarity score.")
        rank_df = pd.DataFrame([
            {"Rank": r, "Doc": doc_ids[i], "Score": result["scores"][i]}
            for r, i in enumerate(result["ranking"], start=1)
        ])
        st.dataframe(
            rank_df, hide_index=True, width="stretch",
            column_config={"Score": st.column_config.NumberColumn(format="%.4f")}
        )

    else:  # Top-k results
        st.markdown(f"The **top {top_k}** ranked documents for this query.")
        max_score = max(result["scores"]) or 1.0
        for rank, idx in enumerate(result["ranking"][:top_k], start=1):
            score = result["scores"][idx]
            with st.container(border=True, gap="xsmall"):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"**#{rank}** &nbsp; :gray[{doc_ids[idx]}]")
                    if rank == 1 and score > 0:
                        st.badge("Top match", icon=":material/star:", color="primary")
                    elif score == 0:
                        st.badge("No overlap", color="gray")
                    st.space("stretch")
                    st.markdown(f"`{score:.4f}`")
                st.markdown(highlight_terms(corpus[idx], query_terms))
                st.progress(min(1.0, score / max_score) if score > 0 else 0.0)


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
    st.caption(f"{QUIZ_LENGTH} questions drawn at random from a bank of {len(QUIZ_QUESTIONS)}. "
               f"Answers are graded when you submit.")

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


# ======================================================================================
# 5. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

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


# Streamlit always renders the navigation widget at the very top of the sidebar, above user content.
# Re-ordering the sidebar's flex children puts the experiment header and progress block above it instead.
SIDEBAR_ORDER_CSS = """
<style>
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarHeader"] { order: 0; }
[data-testid="stSidebarUserContent"] { order: 1; padding-bottom: 0.75rem; }
[data-testid="stSidebarNav"] {
    order: 2;
    border-top: 1px solid rgba(128, 128, 128, 0.28);
    padding-top: 0.5rem;
}
</style>
"""


def render_sidebar():
    with st.sidebar:
        st.markdown(f"### Experiment 4\n**TF-IDF document retrieval**")
        st.caption(f"{EXPERIMENT_CONFIG['course']}  \nRoll nos {EXPERIMENT_CONFIG['roll_no']} · "
                   f"Group {EXPERIMENT_CONFIG['group_no']}")


def render_sidebar_nav(pages: dict):
    """Renders the page links manually, as a list with a sub-list per section.

    st.navigation's built-in sidebar widget always pins itself to the very top of the
    sidebar, so it can't be positioned below other sidebar content. Rendering the links
    ourselves with st.page_link (while st.navigation runs with position="hidden") lets the
    nav sit below the Experiment 4 header instead.
    """
    with st.sidebar:
        for section, section_pages in pages.items():
            if section:
                st.caption(section)
            for p in section_pages:
                st.page_link(p)


def render_sidebar_progress():
    with st.sidebar:
        st.space("small")
        st.markdown("**Your progress**")
        n_trials = len(st.session_state["trials"])
        st.progress(min(n_trials, 3) / 3, text=f"Trials recorded: {n_trials} of 3+")
        if st.session_state.get("quiz_submitted", False):
            score = st.session_state.get("quiz_score", 0)
            st.progress(score / QUIZ_LENGTH, text=f"Quiz: {score} / {QUIZ_LENGTH}")
        else:
            st.badge("Quiz not submitted", icon=":material/schedule:", color="orange")


def main():
    st.set_page_config(
        page_title="TF-IDF Document Retrieval | KGIRS Virtual Lab",
        page_icon=":material/manage_search:",
        layout="wide"
    )

    init_session_state()
    st.html(SIDEBAR_ORDER_CSS)

    page = st.navigation([
        st.Page(render_aim_section, title="Aim", icon=":material/flag:", url_path="aim", default=True),
        st.Page(render_theory_section, title="Theory", icon=":material/menu_book:", url_path="theory"),
        st.Page(render_procedure_section, title="Procedure", icon=":material/checklist:", url_path="procedure"),
        st.Page(render_references_section, title="References", icon=":material/library_books:",
                url_path="references"),
        st.Page(render_simulation_section, title="Simulation", icon=":material/manage_search:", url_path="simulation"),
        st.Page(render_quiz_section, title="Quiz", icon=":material/quiz:", url_path="quiz"),
        st.Page(render_report_section, title="Report", icon=":material/description:", url_path="report"),
    ], position="sidebar")
    page.run()
    render_sidebar_progress()  # after the page so counts include this run's actions


if __name__ == "__main__":
    main()
