"""
Streamlit web app for the question answering system.

Run:  streamlit run app.py
"""

import html
from pathlib import Path

import streamlit as st

from preprocessing.retrieval import load_embedding_model, load_reranker
from preprocessing.text_cleaning import clean_text, extract_pdf_text
from qa_engine import DEFAULT_MODEL, answer_from_document, load_reader

SAMPLE_DOC = Path(__file__).parent / "data" / "documents" / "xv6_sleep_wakeup.txt"

# The app always uses the most accurate model from our evaluation
# (RoBERTa-base, 83 F1 on SQuAD 2.0). Other models, including our own
# fine-tuned DistilBERT, are compared in scripts/evaluate_qa.py.
MODEL_NAME = DEFAULT_MODEL


st.set_page_config(page_title="Question Answering System", layout="wide")

st.markdown("""
<style>
    .block-container { max-width: 1400px; padding: 2rem 3rem; }
    .header {
        padding: 22px 25px;
        border-radius: 12px;
        background: linear-gradient(120deg, #183153, #3267a8);
        margin-bottom: 25px;
    }
    .header h1 { color: white !important; margin: 0; font-size: 2rem; }
    .header p  { color: #e3ecf8 !important; margin: 8px 0 0 0; }
    .answer-box {
        padding: 18px 20px;
        border-radius: 10px;
        border: 1px solid rgba(50, 103, 168, 0.55);
        border-left: 5px solid #3267a8;
        background: rgba(50, 103, 168, 0.12);
        font-size: 1.3rem;
        font-weight: 600;
        overflow-wrap: anywhere;
        white-space: pre-line;
    }
    .answer-box.long { font-size: 1rem; font-weight: 500; line-height: 1.6; }
    .sentence-box {
        margin-top: 10px;
        margin-bottom: 14px;
        padding: 12px 16px;
        border-left: 4px solid #3267a8;
        border-radius: 6px;
        background: rgba(128, 128, 128, 0.10);
        overflow-wrap: anywhere;
    }
    .sentence-box mark { background: #ffe58a; color: #000; padding: 0 2px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the AI models (only once, about 20 seconds)...")
def load_all_models():
    # Loaded once when the app starts and then kept in memory,
    # so every question after that is answered quickly.
    load_reader(MODEL_NAME)
    load_embedding_model()
    load_reranker()
    return True


# Cached so a file is read only once. Without this, Streamlit would re-read
# (and re-run OCR on) the PDF every time the user types something.
@st.cache_data(show_spinner="Reading the document (pages that are images take a few seconds each)...")
def read_file(name, data):
    if name.lower().endswith(".pdf"):
        text = extract_pdf_text(data)
    else:
        text = data.decode("utf-8", errors="replace")
    return clean_text(text)


def read_uploaded_file(file):
    return read_file(file.name, file.getvalue())


st.markdown("""
<div class="header">
    <h1>Question Answering System</h1>
    <p>Upload a document or paste a passage, then ask a question about it.</p>
</div>
""", unsafe_allow_html=True)


# ---------- 1. Input text ----------
st.subheader("1. Add your text")

files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"],
                         accept_multiple_files=True)

use_sample = st.checkbox("Use the sample document (xv6 sleep and wakeup case study)")

passage = st.text_area("Or paste a passage", height=150,
                       placeholder="Paste your text here...")

document_text = ""

for file in files or []:
    try:
        text = read_uploaded_file(file)
        if text:
            document_text += f"\n{text}"
        else:
            st.warning(f"No text found in {file.name}.")
    except Exception as e:
        st.error(f"Could not read {file.name}: {e}")

if use_sample:
    document_text += "\n" + clean_text(SAMPLE_DOC.read_text(encoding="utf-8"))

if passage.strip():
    document_text += "\n" + clean_text(passage)

if document_text.strip():
    with st.expander("Show extracted text"):
        st.text(document_text[:5000])


# ---------- 2. Question ----------
st.subheader("2. Ask a question")

question = st.text_input("Your question",
                         placeholder="e.g. What does the wakeup() function do?")

if st.button("Find Answer", type="primary", use_container_width=True):

    if not document_text.strip() or not question.strip():
        st.warning("Please add some text and type a question.")
        st.stop()

    with st.spinner("Searching the text..."):
        result = answer_from_document(question, document_text, model_name=MODEL_NAME)

    st.subheader("Answer")

    if not result["answer"]:
        st.warning("The answer to this question was not found in the text.")
    else:
        size = " long" if result.get("kind") in ("section", "sentence") else ""
        st.markdown(f'<div class="answer-box{size}">{html.escape(result["answer"])}</div>',
                    unsafe_allow_html=True)

        if result.get("kind") == "section":
            st.caption("Your question asks for a whole section or list, so the "
                       "full section from the document is shown.")
        elif result.get("kind") == "sentence":
            st.caption("No short answer was found, so this is the sentence in the "
                       "document that best matches your question.")
        else:
            st.caption(f"Confidence: {result['score']:.0%}")

        # Show the whole sentence, with the answer highlighted, so answers
        # that are longer than the extracted span (lists, names) are visible.
        sentence = result.get("sentence", "")
        if sentence and sentence != result["answer"]:
            before, _, after = sentence.partition(result["answer"])
            st.markdown(
                '<div class="sentence-box"><b>In the document:</b> '
                f'{html.escape(before)}<mark>{html.escape(result["answer"])}</mark>'
                f'{html.escape(after)}</div>',
                unsafe_allow_html=True
            )

        if result.get("kind") == "span" and result["score"] < 0.10:
            st.info("Low confidence - please check the answer in the document.")

        with st.expander("Where the answer was found"):
            st.write(result["context"])

    if result["retrieved"]:
        with st.expander("Passages that were searched"):
            st.caption("Ranked from most to least relevant to your question.")
            for i, item in enumerate(result["retrieved"], start=1):
                st.markdown(f"**Passage {i}**" + (" (most relevant)" if i == 1 else ""))
                st.write(item["text"])


st.divider()

# Loaded at the END of the page, so the page is shown immediately and the
# user can upload a file / type a question while the models load.
load_all_models()

st.caption(f"Model: {MODEL_NAME} · Built with Hugging Face Transformers and Streamlit")
