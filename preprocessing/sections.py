"""
Split a cleaned document into sections: a heading plus the text under it.

Used for questions whose answer is a whole section or list, e.g.
"display the table of contents", "what are the limitations?",
"who submitted the report?".
"""

import re

# "3.", "4.2", "6.1." at the start of a line
NUMBER_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")

# "8. LIMITATIONS The ..." (a numbered section), not "1. Understand ..." (a list item)
NUMBERED_CAPS_HEADING = re.compile(r"^\d+\.?\s+[A-Z]{2,}\b")


STOP_WORDS = {"the", "of", "a", "an", "and", "in", "on", "for", "to", "by", "is", "are",
               "what", "which", "who", "show", "display", "list", "give", "me", "all",
               "this", "that", "project", "report", "document", "system"}


# Common section names in reports, notes and papers
SECTION_NAMES = {
    "abstract", "introduction", "contents", "objectives", "objective", "motivation",
    "conclusion", "conclusions", "references", "acknowledgement", "acknowledgements",
    "methodology", "results", "result", "discussion", "summary", "background",
    "limitations", "advantages", "disadvantages", "features", "benefits",
    "applications", "scope", "requirements", "overview", "outcomes", "appendix",
}

# "Submitted by: ...", "Academic Year: ..." -> a short label before a colon
LABEL = re.compile(r"^([A-Za-z][A-Za-z ]{2,30}):")


def is_heading(line):
    """A heading line, e.g. "SUBMITTED BY", "TABLE OF CONTENTS", "1.1 Objectives",
    "Table of Contents"."""
    words = line.split()
    letters = [c for c in line if c.isalpha()]
    if not letters or len(words) > 10:
        return False
    mostly_capitals = sum(c.isupper() for c in letters) / len(letters) >= 0.7
    numbered_title = (section_number(line) is not None and len(words) <= 6
                      and not line.endswith((".", ",", ";")))
    named_section = (len(words) <= 5 and not line.endswith((".", ",", ";"))
                     and any(w.lower().strip(":") in SECTION_NAMES for w in words))
    return mostly_capitals or numbered_title or named_section


def heading_words(section):
    """Important words of a section's heading, e.g. {"table", "contents"}."""
    first_line = section.split("\n")[0]
    title = NUMBER_PREFIX.sub("", first_line)
    # For "3. OBJECTIVES The main objectives ..." keep only the capitalised title
    caps = re.match(r"^([A-Z][A-Z0-9&/()\- ]+?)(?=\s+[A-Z][a-z]|$)", title)
    label = LABEL.match(title)
    if caps and len(caps.group(1).split()) <= 10:
        title = caps.group(1)
    elif label:
        title = label.group(1)          # "Submitted by: D.Kundan ..." -> "Submitted by"
    elif section_number(first_line) and title.split()[0].lower() in SECTION_NAMES:
        title = title.split()[0]        # "1.1 Objectives The objectives are ..." -> "Objectives"
    elif not is_heading(first_line):
        return set()
    words = re.findall(r"[a-z]+", title.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def match_heading(question, sections, strict=False):
    """Return the section whose heading words appear in the question, if any.

    "Display the table of contents"  -> TABLE OF CONTENTS
    "Who submitted the report?"      -> SUBMITTED BY
    "List the features of the system"-> 6. FEATURES OF THE PROPOSED SYSTEM

    strict=True is used for ordinary questions ("What is the conclusion?"):
    then every important word of the question must be in the heading, so a
    question like "What is the subject code?" is not mistaken for a section.
    """
    question_words = {w for w in re.findall(r"[a-z]+", question.lower())
                      if w not in STOP_WORDS and len(w) > 2}
    if not question_words:
        return None

    # Sub-sections ("1.1 Objectives ...") can be matched on their own too
    candidates = list(sections)
    for section in sections:
        lines = section.split("\n")
        for i, line in enumerate(lines[1:], start=1):
            if "." in (section_number(line) or ""):
                candidates.append(line)

    best, best_overlap = None, 0.0
    for section in candidates:
        words = heading_words(section)
        if not words:
            continue
        overlap = len(words & question_words) / len(words)
        question_in_heading = question_words <= words
        if strict and not question_in_heading:
            continue
        # "What are the limitations?" -> "3. EXISTING APPROACH AND ITS LIMITATIONS":
        # all the question's words are in the heading, even if the heading is longer
        if question_in_heading:
            overlap = max(overlap, 0.5)
        if overlap >= 0.5 and overlap > best_overlap:
            best, best_overlap = section, overlap
    return best


def section_number(line):
    match = NUMBER_PREFIX.match(line)
    return match.group(1) if match else None


def split_sections(text):
    """Group the paragraphs (one per line) of a cleaned document into sections."""
    sections = []          # list of lists of lines
    current_number = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        number = section_number(line)
        is_subsection = (
            sections and current_number and number
            and number.startswith(current_number + ".")
        )
        only_short_lines = sections and all(len(l.split()) <= 12 for l in sections[-1])

        if is_subsection:
            # "6.2 Estimated Arrival Time ..." belongs to "6. FEATURES ..."
            sections[-1].append(line)
        elif is_heading(line) or LABEL.match(line) or NUMBERED_CAPS_HEADING.match(line):
            # a heading, a "Label: value" line such as "Submitted by: ...",
            # or a new top-level numbered section such as "8. LIMITATIONS ..."
            sections.append([line])
            current_number = number
        elif sections and only_short_lines and is_heading(sections[-1][0]):
            # text right after a heading, e.g. the names after "SUBMITTED BY"
            sections[-1].append(line)
        else:
            sections.append([line])
            current_number = number

    return ["\n".join(lines) for lines in sections]


def format_section(text):
    """Put bullets and numbered items on their own lines for display."""
    if "contents" in text.split("\n")[0].lower():
        # Table of contents: one entry per line ("1. Introduction", "4.2 Define" ...)
        heading, _, body = text.partition("\n")
        body = re.sub(r"\s+(?=\d+(?:\.\d+)*\.?\s+[A-Z])", "\n", body)
        body = re.sub(r"\s+(?=(?:REFERENCES|APPENDIX|ACKNOWLEDGEMENTS?)\b)", "\n", body)
        return f"{heading}\n{body}".strip()

    text = re.sub(r"\s*•\s*", "\n• ", text)
    # numbered list items "1. ", "2. " (1-2 digits, so roll numbers stay put)
    text = re.sub(r"(?<=[.:])\s+(\d{1,2}(?:\.\d+)*\.)\s+(?=[A-Z])", r"\n\1 ", text)
    return text.strip()
