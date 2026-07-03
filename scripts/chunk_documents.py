#!/usr/bin/env python3
"""
Chunk the agricultural knowledge base into passage-sized documents.

Task #5 (RAG correctness epic): the indexer (`scripts/index_cosdata.py`) embeds
**one vector per whole document**, but `intfloat/multilingual-e5-large` truncates
input at **512 tokens**, so any doc longer than that is only partially embedded
(8/47 docs exceed 512 tok, up to 817). Feeding whole docs back to the LLM at
retrieval time also bloats the RAG payload, which tips the larger Amharic prompts
over the 8192-token context window.

This script splits each source document into overlapping, token-bounded passages
and writes them in the **same JSON schema** so the existing indexer and retrieval
path work unchanged. Each chunk becomes its own "document":

    {
        "doc_id":        "<parent_doc_id>#c<n>",   # unique per chunk
        "type":          <parent type>,
        "name":          <parent name>,
        "text":          <chunk text>,
        "source":        <parent source>,
        "parent_doc_id": <parent doc_id>            # new: links chunk -> parent
    }

Docs already under the target size pass through as a single chunk.

Usage (run in the app container so the e5 tokenizer is available):
    docker exec oan_app python scripts/chunk_documents.py \
        --in  assets/all_agricultural_docs.json \
        --out assets/all_agricultural_docs_chunked.json \
        --verify

Tuning (defaults chosen against the real distribution — median 318 tok, max 817):
    --target-tokens 256   target chunk size
    --overlap-tokens 50   tokens of overlap carried between adjacent chunks
    --max-tokens 480      hard cap (< e5's 512) — oversized sentences are split
"""

import os
import re
import sys
import json
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large")


def get_tokenizer():
    """Load the e5 tokenizer so chunk sizes match what the embedder actually sees."""
    from transformers import AutoTokenizer

    # The "passage: " prefix is prepended at embed time; it costs a few tokens, so
    # we budget against the raw text and keep a margin via --max-tokens < 512.
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentence-ish atoms, preferring paragraph and sentence
    boundaries so chunks stay coherent. Newlines and list markers are honoured.
    """
    atoms: List[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        # Split on sentence enders and on single newlines (list items / steps),
        # keeping the delimiter with the preceding sentence.
        parts = re.split(r"(?<=[.!?])\s+|\n+", para)
        for part in parts:
            part = part.strip()
            if part:
                atoms.append(part)
    return atoms


def split_long_atom(tokenizer, atom: str, max_tokens: int) -> List[str]:
    """Hard-split a single atom that exceeds max_tokens, on word boundaries."""
    words = atom.split()
    pieces: List[str] = []
    cur: List[str] = []
    for word in words:
        cur.append(word)
        if count_tokens(tokenizer, " ".join(cur)) > max_tokens:
            cur.pop()
            if cur:
                pieces.append(" ".join(cur))
            cur = [word]
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_text(
    tokenizer,
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
) -> List[str]:
    """
    Greedily pack sentence atoms into chunks of ~target_tokens (hard cap max_tokens),
    carrying ~overlap_tokens of trailing sentences into the next chunk for continuity.
    """
    atoms = split_sentences(text)

    # Pre-split any atom that alone exceeds the hard cap.
    normalized: List[str] = []
    for atom in atoms:
        if count_tokens(tokenizer, atom) > max_tokens:
            normalized.extend(split_long_atom(tokenizer, atom, max_tokens))
        else:
            normalized.append(atom)

    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    for atom in normalized:
        atom_tokens = count_tokens(tokenizer, atom)
        # Close the current chunk if adding this atom would push us over target,
        # but only once we have something to emit.
        if cur and cur_tokens + atom_tokens > target_tokens:
            chunks.append(" ".join(cur))
            # Build the overlap tail from the end of the chunk we just emitted.
            tail: List[str] = []
            tail_tokens = 0
            for prev in reversed(cur):
                t = count_tokens(tokenizer, prev)
                if tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            cur = tail
            cur_tokens = tail_tokens
        cur.append(atom)
        cur_tokens += atom_tokens

    if cur:
        chunks.append(" ".join(cur))

    return chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    tokenizer,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in documents:
        parent_id = doc.get("doc_id", "")
        text = doc.get("text", "") or ""
        passages = chunk_text(tokenizer, text, target_tokens, overlap_tokens, max_tokens)
        if not passages:
            passages = [text]
        for n, passage in enumerate(passages):
            out.append(
                {
                    "doc_id": f"{parent_id}#c{n}" if len(passages) > 1 else parent_id,
                    "type": doc.get("type", "document"),
                    "name": doc.get("name", ""),
                    "text": passage,
                    "source": doc.get("source", ""),
                    "parent_doc_id": parent_id,
                }
            )
    return out


def load_documents(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "documents" in data:
        return data["documents"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized document JSON shape in {path}")


def report(tokenizer, chunks: List[Dict[str, Any]], max_tokens: int) -> None:
    counts = [count_tokens(tokenizer, c["text"]) for c in chunks]
    over = [c for c in counts if c > 512]
    print("\n=== Chunk distribution ===")
    print(f"chunks: {len(chunks)}")
    print(
        f"tokens: min {min(counts)} | median {int(statistics.median(counts))} | "
        f"mean {int(statistics.mean(counts))} | max {max(counts)}"
    )
    print(f"chunks over 512 tok (would truncate): {len(over)}")
    if over:
        print(f"  -> WARNING oversized chunks: {sorted(over, reverse=True)}")
    parents = {c["parent_doc_id"] for c in chunks}
    print(f"parents: {len(parents)} | avg chunks/parent: {len(chunks) / len(parents):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="infile", default="assets/all_agricultural_docs.json")
    parser.add_argument("--out", dest="outfile", default="assets/all_agricultural_docs_chunked.json")
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=480)
    parser.add_argument("--verify", action="store_true", help="Print a sample chunk per first few parents")
    args = parser.parse_args()

    in_path = (project_root / args.infile) if not os.path.isabs(args.infile) else Path(args.infile)
    out_path = (project_root / args.outfile) if not os.path.isabs(args.outfile) else Path(args.outfile)

    documents = load_documents(in_path)
    print(f"Loaded {len(documents)} source documents from {in_path}")

    tokenizer = get_tokenizer()
    chunks = chunk_documents(
        documents, tokenizer, args.target_tokens, args.overlap_tokens, args.max_tokens
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"documents": chunks}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(chunks)} chunks to {out_path}")

    report(tokenizer, chunks, args.max_tokens)

    if args.verify:
        print("\n=== Sample chunks ===")
        seen = set()
        for c in chunks:
            p = c["parent_doc_id"]
            if p in seen:
                continue
            seen.add(p)
            preview = c["text"][:200].replace("\n", " ")
            print(f"[{c['doc_id']}] {preview}...")
            if len(seen) >= 3:
                break


if __name__ == "__main__":
    main()
