"""Fast unit tests (no model download needed).  Run:  python -m pytest"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_metrics, normalize_answer, score_example  # noqa: E402
from evaluation.qa_datasets import load_custom, load_document_qa  # noqa: E402
from preprocessing.chunking import split_text  # noqa: E402
from preprocessing.text_cleaning import clean_text  # noqa: E402


class WordTokenizer:
    """Tiny stand-in for a real tokenizer: one token per word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


# ---- metrics ---------------------------------------------------------------

def test_normalize_removes_case_punctuation_and_articles():
    assert normalize_answer("The  Computer-Science dept.") == "computerscience dept"


def test_exact_and_partial_match():
    assert score_example("make qemu", ["make qemu"]) == (1, 1.0)
    em, f1 = score_example("30", ["30 days"])
    assert em == 0 and round(f1, 3) == 0.667


def test_unanswerable_questions():
    assert score_example("", []) == (1, 1.0)
    assert score_example("something", []) == (0, 0.0)
    assert score_example("", ["an answer"]) == (0, 0.0)


def test_compute_metrics_groups():
    m = compute_metrics(["a b", "", "x"], [["a b"], [], ["y"]])
    assert m["exact_match"] == round(100 * 2 / 3, 2)
    assert m["has_answer_count"] == 2 and m["no_answer_count"] == 1
    assert m["no_answer_f1"] == 100.0


# ---- preprocessing ---------------------------------------------------------

def test_clean_text_joins_lines_and_removes_toc_dots():
    raw = "Submitted by:\n\nAlice Smith\nBob Jones\n\n1. INTRODUCTION ........ 5\n"
    cleaned = clean_text(raw)
    assert "Submitted by: Alice Smith Bob Jones" in cleaned
    assert "...." not in cleaned


def test_clean_text_fixes_hyphenated_words():
    assert clean_text("synchroni-\nzation works") == "synchronization works"


def test_split_text_respects_chunk_size():
    text = "\n".join(f"paragraph {i} " + "word " * 20 for i in range(10))
    chunks = split_text(text, WordTokenizer(), chunk_size=50)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 50 for c in chunks)
    # no text is lost
    assert sum(len(c.split()) for c in chunks) == len(text.split())


def test_keyword_scores_with_only_stop_words():
    from preprocessing.retrieval import keyword_scores
    assert list(keyword_scores(["the is a", "an the"], "the")) == [0.0, 0.0]


def test_question_cleaning_and_rephrasing():
    from qa_engine import clean_question, rephrase_question
    assert clean_question("  the report is submitted by whom?? ") == "The report is submitted by whom?"
    assert rephrase_question("The report was prepared by whom?") == "Who prepared the report?"
    assert rephrase_question("What is a file?") is None


def test_sentence_splitting_keeps_abbreviations_and_headings():
    from qa_engine import split_sentences
    text = "3. N. Sai Deepanvitha (Roll No. 298)\nSet up xv6 using QEMU. Tested it."
    assert split_sentences(text) == [
        "3. N. Sai Deepanvitha (Roll No. 298) Set up xv6 using QEMU.",
        "Tested it.",
    ]


def test_descriptive_question_detection():
    from qa_engine import DESCRIPTIVE_QUESTION
    assert DESCRIPTIVE_QUESTION.match("What did N. Sai Deepanvitha do?")
    assert DESCRIPTIVE_QUESTION.match("Explain busy waiting?")
    assert not DESCRIPTIVE_QUESTION.match("What is the subject code?")


def test_sections_and_heading_match():
    from preprocessing.sections import format_section, match_heading, split_sections
    text = ("SUBMITTED BY\nA. Rao\nB. Kumar\nTABLE OF CONTENTS\n"
            "Abstract 3 1. Introduction 4 2. Problem Statement 4 3. Objectives 5\n"
            "6. FEATURES OF THE SYSTEM\n6.1 Tracking shows the bus.\n6.2 Alerts warn about delays.\n"
            "8. LIMITATIONS The system has limits: 1. Needs GPS. 2. Needs internet.")
    sections = split_sections(text)
    assert sections[0] == "SUBMITTED BY\nA. Rao\nB. Kumar"
    assert "6.2 Alerts" in sections[2]
    assert match_heading("Who submitted the report?", sections) == sections[0]
    assert match_heading("Display the table of contents", sections) == sections[1]
    assert match_heading("What is the subject code?", sections, strict=True) is None
    assert format_section(sections[3]).endswith("limits:\n1. Needs GPS.\n2. Needs internet.")


def test_page_footers_are_removed():
    assert clean_text("Some text here\n\nPage 3\n\nMore text") == "Some text here\nMore text"


def test_split_text_cuts_very_long_paragraph():
    chunks = split_text("word " * 120, WordTokenizer(), chunk_size=50)
    assert [len(c.split()) for c in chunks] == [50, 50, 20]


# ---- datasets --------------------------------------------------------------

def test_custom_dataset_answers_are_in_context():
    examples = load_custom()  # raises if any answer is not copied exactly
    assert len(examples) >= 50
    assert any(not ex["answers"] for ex in examples)  # has unanswerable questions


def test_document_dataset_loads():
    examples = load_document_qa()
    assert len(examples) >= 15
    assert "sleep" in examples[0]["context"].lower()
