"""Core question answering logic, shared by the web app and the evaluation
scripts so both measure exactly the same system.

Pipeline:
    document text
      -> split into chunks                (preprocessing/chunking.py)
      -> find the most relevant chunks    (preprocessing/retrieval.py)
      -> extract the answer span with a   (transformer reader model)
         BERT-style QA model
      -> if no short answer is found:     best matching sentence, chosen by
                                          the cross-encoder re-ranker
"""

import re
from functools import lru_cache

import numpy as np
from transformers import pipeline

from preprocessing.chunking import split_text
from preprocessing.retrieval import load_reranker, retrieve_relevant_chunks
from preprocessing.sections import format_section, match_heading, split_sections

DEFAULT_MODEL = "deepset/roberta-base-squad2"

# Cross-encoder score above which a sentence counts as answering the question.
# Chosen from tests on our document: questions the text answers scored +2.9 to
# +8.2, unrelated questions ("capital of France") scored -1 to -11.
SENTENCE_THRESHOLD = 1.0

# A heading glued to the start of a sentence, e.g. "4.2 SLEEP MECHANISM The ..."
LEADING_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:[A-Z][A-Z0-9()&,/-]*\s+)+(?=[A-Z][a-z])")

# Text ending in an abbreviation, where a "." does NOT end the sentence:
# initials ("N."), "No.", "Dr.", "e.g.", numbered items ("3.") ...
ENDS_WITH_ABBREVIATION = re.compile(
    r"(?:\b[A-Za-z]|\bNo|\bDr|\bMr|\bMrs|\bMs|\bFig|\bvs|\betc|\be\.g|\bi\.e|^\d+(?:\.\d+)*)\.$"
)

# Requests whose answer is a whole section or list
SECTION_QUESTION = re.compile(
    r"^(?:display|show|list|give|print|enumerate|tell\s+me|name\s+the|summari[sz]e|"
    r"what\s+are|which\s+are|who\s+(?:submitted|prepared|wrote|made|created|developed|are)\b)"
    r"|\bby\s+whom\b"
    r"|\btable\s+of\s+contents?\b|\b(?:objectives|features|benefits|advantages|disadvantages|"
    r"limitations|stages|steps|types|members)\b",
    re.IGNORECASE,
)

# Questions that ask for an explanation rather than a short fact
DESCRIPTIVE_QUESTION = re.compile(
    r"^(?:what\s+(?:does|do|did)\s+.+\s+do\b|explain|describe|why\b|how\s+(?:does|do|did|is|are|can)\b)",
    re.IGNORECASE,
)


@lru_cache(maxsize=2)
def load_reader(model_name=DEFAULT_MODEL):
    """Load a Hugging Face question-answering model (cached)."""
    return pipeline("question-answering", model=model_name)


def answer_from_passage(question, passage, model_name=DEFAULT_MODEL):
    """Answer a question from one short passage (no retrieval step).

    Returns the answer text, or "" if the model thinks the passage does
    not contain the answer.
    """
    reader = load_reader(model_name)
    result = reader(
        question=question,
        context=passage,
        handle_impossible_answer=True,
        max_answer_len=50,
    )
    return result["answer"].strip(), float(result["score"])


PASSIVE_WHOM = re.compile(
    r"^(?:the\s+)?(?P<thing>.+?)\s+(?:is|was|are|were|has been|have been)\s+"
    r"(?P<verb>\w+)\s+by\s+whom\W*$",
    re.IGNORECASE,
)


def clean_question(question):
    """Tidy the question: single spaces, one question mark at the end."""
    question = " ".join(question.split()).rstrip("?!. ")
    return question[:1].upper() + question[1:] + "?" if question else ""


def rephrase_question(question):
    """Rewrite passive "by whom" questions, which QA models handle badly.

    "the report was prepared by whom" -> "Who prepared the report?"
    Returns None if no rule applies.
    """
    match = PASSIVE_WHOM.match(question.rstrip("?"))
    if not match:
        return None
    return f"Who {match['verb']} the {match['thing']}?"


def answer_from_document(question, text, model_name=DEFAULT_MODEL, top_k=5):
    """Answer a question from a document.

    1. The reader model looks for a short answer span.
    2. If none is found, retry once with a rephrased question.
    3. If still none, return the single most relevant sentence, but only if
       the re-ranker is confident it answers the question.
    """
    question = clean_question(question)

    # "Display the table of contents", "What are the limitations?",
    # "Who submitted the report?": the answer is a whole section / list.
    # "What is the conclusion?": a question that just names a section heading.
    is_section_request = bool(SECTION_QUESTION.search(question))
    section = best_section(question, text, heading_only=not is_section_request)
    if section:
        return section

    result = _answer_from_document(question, text, model_name, top_k)

    # "What did X do?", "Explain ...", "Why ...": a full sentence is a better
    # answer than the short phrase the reader model extracts.
    if DESCRIPTIVE_QUESTION.match(question) and result["retrieved"]:
        sentence, logit = best_sentence(question, result["retrieved"])
        if sentence and logit >= SENTENCE_THRESHOLD:
            return _sentence_result(result, sentence, logit)

    if not result["answer"]:
        rephrased = rephrase_question(question)
        if rephrased:
            retry = _answer_from_document(rephrased, text, model_name, top_k)
            if retry["answer"]:
                return retry

    if not result["answer"] and result["retrieved"]:
        sentence, logit = best_sentence(question, result["retrieved"])
        if sentence and logit >= SENTENCE_THRESHOLD:
            return _sentence_result(result, sentence, logit)

    return result


def best_section(question, text, heading_only=False):
    """Return the document section that best answers a list/section request,
    or None if no section matches well enough.

    heading_only=True (ordinary questions): only return a section if the
    question is just its heading, e.g. "What is the conclusion?".
    """
    sections = split_sections(text)
    if not sections:
        return None

    # 1. A section whose heading is named in the question
    by_heading = match_heading(question, sections, strict=heading_only)
    if heading_only and not by_heading:
        return None
    if by_heading:
        return {
            "answer": format_section(by_heading),
            "kind": "section",
            "score": 1.0,
            "context": by_heading,
            "sentence": "",
            "retrieved": [{"text": by_heading, "score": 1.0}],
        }

    # 2. Otherwise the section the cross-encoder finds most relevant
    top = retrieve_relevant_chunks(sections, question, top_k=1)
    if not top:
        return None

    # retrieve_relevant_chunks returns the cross-encoder score through a
    # sigmoid; convert the threshold the same way.
    if top[0]["score"] < 1 / (1 + np.exp(-SENTENCE_THRESHOLD)):
        return None

    return {
        "answer": format_section(top[0]["text"]),
        "kind": "section",
        "score": top[0]["score"],
        "context": top[0]["text"],
        "sentence": "",
        "retrieved": top,
    }


def _sentence_result(result, sentence, logit):
    context = next((r["text"] for r in result["retrieved"] if sentence in r["text"]),
                   result["retrieved"][0]["text"])
    return {
        **result,
        "answer": sentence,
        "kind": "sentence",
        "score": float(1 / (1 + np.exp(-logit))),
        "sentence": "",
        "context": context,
    }


def split_sentences(text):
    """Split text into sentences.

    Unlike a plain split on ". ", this does not break after abbreviations
    ("N. Sai", "Roll No. 298"), and a short heading line without a full stop
    (e.g. "3. N. Sai Deepanvitha (Roll No. 298)") is joined to the sentence
    after it, so the heading and its description stay together.
    """
    sentences = []
    for line in text.split("\n"):
        buffer = ""
        for part in re.split(r"(?<=[.!?])\s+", line.strip()):
            buffer = f"{buffer} {part}".strip()
            if buffer and not ENDS_WITH_ABBREVIATION.search(buffer):
                sentences.append(buffer)
                buffer = ""
        if buffer:
            sentences.append(buffer)

    merged = []
    for sentence in sentences:
        previous = merged[-1] if merged else ""
        is_heading = previous and len(previous.split()) <= 10 and not previous.endswith((".", "?", "!"))
        if is_heading:
            merged[-1] = f"{previous} {sentence}"
        else:
            merged.append(sentence)
    return merged


def best_sentence(question, retrieved):
    """Pick the sentence from the retrieved chunks that best answers the question."""
    sentences = []
    for item in retrieved:
        for sentence in split_sentences(item["text"]):
            sentence = LEADING_HEADING.sub("", sentence)
            if len(sentence.split()) >= 4 and sentence not in sentences:
                sentences.append(sentence)

    if not sentences:
        return "", float("-inf")

    scores = load_reranker().predict([(question, s) for s in sentences])
    best = int(np.argmax(scores))
    return sentences[best], float(scores[best])


def _answer_from_document(question, text, model_name=DEFAULT_MODEL, top_k=5):
    """Answer a question from a document of any length.

    Returns a dict:
        answer     - answer text ("" when not found)
        score      - reader confidence for the chosen answer
        context    - the chunk the answer came from
        sentence   - the full sentence / line containing the answer
        retrieved  - the chunks that were searched, with similarity scores
    """
    question = question.strip()
    empty = {"answer": "", "kind": "", "score": 0.0, "context": "", "sentence": "", "retrieved": []}

    if not question or not text.strip():
        return empty

    reader = load_reader(model_name)
    chunks = split_text(text, reader.tokenizer)
    retrieved = retrieve_relevant_chunks(chunks, question, top_k=top_k)

    candidates = []
    for item in retrieved:
        result = reader(
            question=question,
            context=item["text"],
            handle_impossible_answer=True,
            max_answer_len=50,
        )
        # A real answer never runs across two paragraphs or a heading,
        # e.g. "22CSC15N\nPROBLEM STATEMENT" -> keep "22CSC15N"
        answer = next(
            (line.strip() for line in result["answer"].splitlines() if line.strip()),
            ""
        )

        # An empty answer means "this chunk does not contain the answer"
        if answer:
            start = item["text"].find(answer, max(0, result["start"] - 5))
            candidates.append({
                "answer": answer,
                "kind": "span",
                "score": float(result["score"]),
                "context": item["text"],
                "sentence": sentence_around(item["text"], start, len(answer)),
            })

    if not candidates:
        return {**empty, "sentence": "", "retrieved": retrieved}

    best = max(candidates, key=lambda c: c["score"])
    return {**best, "retrieved": retrieved}


def sentence_around(text, start, length):
    """Return the full sentence (or line) that contains the answer.

    The model returns a short span. Showing the whole sentence helps when the
    real answer is longer, e.g. a list of names after "Submitted by:".
    """
    if start < 0:
        return ""
    end = start + length

    left = max(text.rfind("\n", 0, start), text.rfind(". ", 0, start))
    left = 0 if left < 0 else left + 1

    candidates = [i for i in (text.find("\n", end), text.find(". ", end)) if i >= 0]
    right = min(candidates) + 1 if candidates else len(text)

    return text[left:right].strip()
