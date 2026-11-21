"""Split long documents into chunks the QA model can read.

Transformer readers can only look at ~512 tokens at a time, so a long
document is cut into smaller pieces. We cut on paragraph boundaries where
possible so sentences are not broken in the middle.
"""

# 250 tokens per chunk gave the best accuracy in our tests
# (smaller chunks lose context, bigger ones mix several topics).
DEFAULT_CHUNK_SIZE = 250


def split_text(text, tokenizer, chunk_size=DEFAULT_CHUNK_SIZE):
    """Return a list of text chunks of at most `chunk_size` tokens."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    chunks = []
    current_lines = []
    current_tokens = 0

    for line in lines:

        line_tokens = len(tokenizer.encode(line, add_special_tokens=False))

        # A single paragraph longer than the chunk size is cut by tokens
        if line_tokens > chunk_size:

            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_tokens = 0

            tokens = tokenizer.encode(line, add_special_tokens=False)

            for start in range(0, len(tokens), chunk_size):
                part = tokens[start:start + chunk_size]
                chunk = tokenizer.decode(part, skip_special_tokens=True)
                if chunk.strip():
                    chunks.append(chunk.strip())

            continue

        # Start a new chunk when the size limit would be exceeded
        if current_tokens + line_tokens > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_tokens = 0

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks
