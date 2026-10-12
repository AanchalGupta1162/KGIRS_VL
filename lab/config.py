"""Static experiment content: lab copy, theory/application text, the quiz bank, and the
default corpus/query/weighting-scheme options. No Streamlit calls here - this is data only.
"""
from pathlib import Path

EXPERIMENT_CONFIG = {
    "title": "Experiment 4: TF-IDF Based Document Retrieval",
    "course": "Knowledge Graphs and Information Retrieval Systems",
    "roll_no": "16-20",
    "group_no": "4",
    "aim": "To represent a document collection and a search query as **TF-IDF weighted vectors** in the "
           "vector space model, rank the documents by their **cosine similarity** to the query, observe how "
           "the choice of TF and IDF weighting scheme changes that ranking.",
    
}



THEORY_CONTENT = {
    
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
    {
        "name": "E-commerce product search",
        "icon": ":material/shopping_bag:",
        "doc": "A product catalogue",
        "query_is": "A shopper's search",
        "metric": "nDCG@k",
        "why": "Common words such as *wireless* appear everywhere. Specific terms such as *noise-cancelling* "
               "push the most useful products to the top.",
        "query": "wireless noise cancelling headphones for travel",
        "corpus": [
            "Over-ear wireless noise cancelling headphones with a foldable design, long battery life and a travel case.",
            "Compact wired studio headphones with balanced audio, a detachable cable and a padded headband.",
            "Wireless earbuds with a charging case, sweat resistance and a secure fit for running and workouts.",
            "Lightweight travel backpack with a laptop sleeve, water-resistant fabric and multiple organiser pockets.",
            "Premium noise cancelling headphones with an adjustable headband, microphone and touch controls.",
        ],
    },
]



APPLICATION_EXPLANATIONS = {
    "Web and site search": "Each page is a document and the visitor's search is the query. TF counts how often "
    "query terms appear on each page, while IDF gives more weight to specific terms such as *reset* than to "
    "common terms such as *account*. Cosine similarity ranks the pages most likely to answer the search first.",
    "Spam and phishing filtering": "Each message is a document and a message is compared with a profile of "
    "known spam terms. Words shared by many messages contribute less, while distinctive terms such as *urgent*, "
    "*suspended* and *verify* receive more weight. The highest-scoring messages can then be sent to a spam queue.",
    "More-like-this recommendation": "The article being read becomes the query and every catalogue article is a "
    "document. TF-IDF represents each article by its important terms, reducing the influence of broad words such "
    "as *search*. Cosine similarity finds articles with a similar topic even when their lengths differ.",
    "Plagiarism and duplicate detection": "The submitted passage is the query and earlier submissions are the "
    "documents. TF-IDF highlights unusual phrases shared by both texts, while common academic words receive less "
    "weight. A high cosine score flags passages that deserve a closer human comparison.",
    "Resume and job matching": "The job description is the query and each resume is a document. Common words "
    "such as *experience* contribute little when they occur in many resumes, while skills such as *Python* and "
    "*deployment* distinguish candidates. The ranked list helps a recruiter review the closest matches first.",
    "Support ticket routing": "Each support queue is represented by a document containing the issues it handles, "
    "and an incoming ticket is the query. TF-IDF gives a strong signal to terms that identify one queue, such as "
    "*invoice* and *tax* for billing. The highest-scoring queue becomes the suggested destination.",
    "E-commerce product search": "Each product description is a document and the shopper's words form the query. "
    "TF measures how strongly a product mentions a search term; IDF prevents generic terms such as *wireless* "
    "from dominating when they appear everywhere. Cosine similarity ranks the products that best match the full "
    "request, including distinctive terms such as *noise cancelling* and *travel*.",
}

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
                "rare *elsewhere*. "
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


TF_SHORT = {
    "Raw Term Frequency": "Raw count",
    "Normalized Term Frequency": "Normalized",
    "Log-Normalized Term Frequency": "Log-normalized",
}
IDF_SHORT = {
    "Standard IDF: log(N / df)": "Standard",
    "Smoothed IDF: log(1 + N / df)": "Smoothed",
}

# How many questions a student is asked in one sitting. The bank below holds exactly this many,
# so every sitting covers the whole bank in a freshly shuffled order.
QUIZ_LENGTH = 10


# The question bank. Four of these require working out a number by hand (tf, idf or precision@k)
# rather than just recalling a definition. Option order is fixed because options are labelled A) to D).
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
        "question": "A corpus has N = 100 documents. A term appears in df = 10 of them. Using the "
                    "standard formula idf = log10(N / df), what is its IDF?",
        "options": [
            "A) 0.5",
            "B) 1.0",
            "C) 2.0",
            "D) 10.0",
        ],
        "answer_index": 1,
        "explanation": "idf = log10(100 / 10) = log10(10) = 1.0. Work the ratio out first, then take the "
                       "log10 of that ratio - the log is what keeps the weight from growing with N or df "
                       "directly.",
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
        "question": "A document contains 25 tokens in total, and the term 'graph' appears 5 times. Using "
                    "normalized term frequency = count(t,d) / |d|, what is TF for 'graph' in this document?",
        "options": [
            "A) 0.05",
            "B) 0.2",
            "C) 5.0",
            "D) 25.0",
        ],
        "answer_index": 1,
        "explanation": "tf = count / document length = 5 / 25 = 0.2. Dividing by the document's token "
                       "count is what stops a document from scoring higher simply for being longer.",
    },
    {
        "id": 9,
        "question": "A search returns 5 documents at k = 5. Of those, 3 are judged relevant. What is "
                    "Precision@5?",
        "options": [
            "A) 0.3",
            "B) 0.5",
            "C) 0.6",
            "D) 1.67",
        ],
        "answer_index": 2,
        "explanation": "Precision@k = relevant retrieved / k = 3 / 5 = 0.6. It only looks at what was "
                       "returned, not at how many relevant documents exist in total - that is recall's job.",
    },
    {
        "id": 10,
        "question": "A term appears 9 times in a document. Using log-normalised TF = 1 + log10(count(t,d)), "
                    "what is its TF, to two decimal places?",
        "options": [
            "A) 0.95",
            "B) 1.95",
            "C) 9.00",
            "D) 2.20",
        ],
        "answer_index": 1,
        "explanation": "tf = 1 + log10(9) = 1 + 0.95 = 1.95. Nine repetitions barely cost more than one "
                       "would, which is exactly the point of compressing counts with a logarithm.",
    },
]

# Default sample corpus: 41 documents, one per line, across five course topics (information retrieval,
# knowledge graphs, machine learning / NLP, databases, semantic search / RAG) plus two long documents.
SAMPLE_CORPUS_PATH = Path(__file__).resolve().parent.parent / "sample_corpus.txt"
DEFAULT_CORPUS = [line.strip() for line in SAMPLE_CORPUS_PATH.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
DEFAULT_QUERY = "TF-IDF document ranking"

DEFAULT_NOTES = ("The TF-IDF retrieval trials demonstrated consistent, query-relevant document ranking across "
                 "the tested weighting-scheme configurations.")

# Sample queries that work well against the built-in corpus, offered as quick picks in the sidebar.
# The document lists are kept as a record of which documents are on-topic for each query.
