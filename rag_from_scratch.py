#!/usr/bin/env python3
# rag_from_scratch.py
"""
Minimal, interview-ready RAG (Retrieval-Augmented Generation) pipeline
implemented from scratch (no sklearn). Single-file, pure Python + NumPy.

Components:
1) Tokenization + simple chunking
2) TF-IDF embeddings (implemented manually)
3) Cosine-similarity retriever (top-k)
4) Prompt construction with citations
5) Tiny "LLM" stub (extractive) you can replace with a real LLM call

Run:
    python rag_from_scratch.py

Author: (you)
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------
# 0) Dummy corpus
# ----------------------------
DOCS: List[Tuple[str, str]] = [
    ("doc1", "Logistic regression is a linear model used for binary classification. "
             "It uses the sigmoid function to map scores to probabilities."),
    ("doc2", "Gradient descent updates parameters in the opposite direction of the gradient "
             "of the loss function with respect to the parameters."),
    ("doc3", "Retrieval-Augmented Generation (RAG) retrieves documents relevant to a query and "
             "feeds them into a language model as context."),
    ("doc4", "TF-IDF stands for term frequency–inverse document frequency and is a classical "
             "approach to text retrieval."),
    ("doc5", "Regularization such as L2 can help prevent overfitting by penalizing large weights "
             "in linear models.")
]


# ----------------------------
# 1) Tokenization + Chunking
# ----------------------------
_word_re = re.compile(r"[a-zA-Z0-9]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _word_re.findall(text)]

def simple_chunk(text: str, max_tokens: int = 80) -> List[str]:
    toks = tokenize(text)
    if not toks:
        return []
    chunks = []
    for i in range(0, len(toks), max_tokens):
        chunk_tokens = toks[i:i+max_tokens]
        chunks.append(" ".join(chunk_tokens))
    return chunks

@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str

def build_chunks(docs: List[Tuple[str,str]], max_tokens: int = 80) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc_id, text in docs:
        for j, ch in enumerate(simple_chunk(text, max_tokens=max_tokens)):
            all_chunks.append(Chunk(doc_id=doc_id, chunk_id=j, text=ch))
    return all_chunks


# ----------------------------
# 2) TF-IDF (manual implementation)
# ----------------------------
class TfidfVectorizerScratch:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self._fitted = False

    def _build_vocab(self, docs: List[List[str]]):
        vocab = {}
        for tokens in docs:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab

    def _compute_idf(self, docs: List[List[str]]):
        N = len(docs)
        df = np.zeros(len(self.vocab), dtype=np.float64)
        for tokens in docs:
            seen = set(tokens)
            for t in seen:
                df[self.vocab[t]] += 1.0
        # smooth idf: log((N + 1) / (df + 1)) + 1
        self.idf = np.log((N + 1.0) / (df + 1.0)) + 1.0

    def fit(self, texts: List[str]):
        tokenized = [tokenize(t) for t in texts]
        self._build_vocab(tokenized)
        self._compute_idf(tokenized)
        self._fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        assert self._fitted, "Vectorizer not fitted"
        V = len(self.vocab)
        X = np.zeros((len(texts), V), dtype=np.float64)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            tf = {}
            for t in tokens:
                if t in self.vocab:
                    tf[t] = tf.get(t, 0) + 1
            # l2 normalize tf-idf row after applying idf
            for t, cnt in tf.items():
                j = self.vocab[t]
                X[i, j] = (cnt / len(tokens)) * self.idf[j]
            # l2 normalization
            norm = np.linalg.norm(X[i])
            if norm > 0:
                X[i] /= norm
        return X

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


# ----------------------------
# 3) Retriever (cosine similarity)
# ----------------------------
def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: (m, d), B: (n, d)
    # assume rows already l2-normalized; safeguard otherwise
    def safe_norm_rows(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return M / norms
    A_n = safe_norm_rows(A)
    B_n = safe_norm_rows(B)
    return A_n @ B_n.T

class Retriever:
    def __init__(self, chunks: List[Chunk], embedder: TfidfVectorizerScratch):
        self.chunks = chunks
        self.embedder = embedder
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_vecs = self.embedder.fit_transform(self.chunk_texts)  # (C, V)

    def search(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        q_vec = self.embedder.transform([query])  # (1, V)
        sims = cosine_similarity_matrix(q_vec, self.chunk_vecs)[0]  # (C,)
        top_idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]


# ----------------------------
# 4) Prompt builder
# ----------------------------
SYS_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "Cite sources like [doc_id#chunk_id]. If the answer isn't in context, say you don't know."
)

def build_prompt(query: str, contexts: List[Tuple[Chunk, float]]) -> str:
    ctx_blocks = []
    for ch, score in contexts:
        ctx_blocks.append(
            f"[{ch.doc_id}#{ch.chunk_id}] (score={score:.3f})\n{ch.text}"
        )
    context_str = "\n\n".join(ctx_blocks)
    prompt = (
        f"{SYS_PROMPT}\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )
    return prompt


# ----------------------------
# 5) “LLM” stub (extractive baseline)
# ----------------------------
def cheap_extractive_llm(prompt: str) -> str:
    """
    Extremely simple generator:
    - Pulls sentences from CONTEXT containing question keywords
    - Emits them and keeps citations included in the context block labels
    Replace with your real LLM call when integrating.
    """
    # Extract context region
    m = re.search(r"CONTEXT:\n(.*?)\n\nQUESTION:", prompt, flags=re.S)
    context = m.group(1) if m else ""
    # Extract question
    q = re.search(r"QUESTION:(.*?)\n\nANSWER:", prompt, flags=re.S)
    question = q.group(1).strip() if q else ""

    q_terms = [t for t in tokenize(question) if len(t) > 2]
    # Split context blocks and sentences
    blocks = context.split("\n\n")
    picked: List[str] = []
    cites: List[str] = []

    for blk in blocks:
        # citation label is on the first line like: [docX#0] (score=...)
        m2 = re.match(r"\[(\w+?#\d+)\]", blk)
        cite = m2.group(1) if m2 else None
        # sentences
        sents = re.split(r"(?<=[.!?])\s+", blk)
        for s in sents:
            score = sum(s.count(t) for t in q_terms)
            if score > 0:
                picked.append(s.strip())
                if cite:
                    cites.append(cite)

    if not picked:
        return "I don't know based on the provided context."

    cite_str = ", ".join(sorted(set(cites)))
    # Simple "summary": join the top lines
    text = " ".join(picked[:3]).strip()
    if cite_str:
        text += f" [{cite_str}]"
    return text


# ----------------------------
# 6) RAG pipeline (end-to-end)
# ----------------------------
class RAG:
    def __init__(self, docs: List[Tuple[str,str]], chunk_max_tokens: int = 80):
        self.chunks = build_chunks(docs, max_tokens=chunk_max_tokens)
        self.retriever = Retriever(self.chunks, TfidfVectorizerScratch())

    def answer(self, query: str, k: int = 3, generator=cheap_extractive_llm) -> Dict:
        hits = self.retriever.search(query, k=k)
        prompt = build_prompt(query, hits)
        answer = generator(prompt)
        return {
            "query": query,
            "retrieved": [
                {"doc_id": ch.doc_id, "chunk_id": ch.chunk_id, "score": score, "text": ch.text}
                for ch, score in hits
            ],
            "prompt": prompt,
            "answer": answer,
        }


# ----------------------------
# 7) Demo
# ----------------------------
def main():
    rag = RAG(DOCS, chunk_max_tokens=80)

    queries = [
        "What is RAG and how does it help language models?",
        "Explain logistic regression briefly.",
        "How does gradient descent update parameters?",
        "What is TF-IDF?",
        "How can regularization help linear models?",
        "What is softmax? (not in corpus)",
    ]

    for q in queries:
        out = rag.answer(q, k=3)
        print("\n=== QUESTION ===")
        print(q)
        print("\n--- ANSWER ---")
        print(out["answer"])
        print("\n--- RETRIEVED CONTEXTS ---")
        for r in out["retrieved"]:
            print(f"- [{r['doc_id']}#{r['chunk_id']}] score={r['score']:.3f} :: {r['text']}")

if __name__ == "__main__":
    main()
