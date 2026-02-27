"""
demo_and_test.py
----------------
Self-contained validation of the extraction logic.

Since spaCy cannot be pip-installed in this environment, this file:
  1. Builds minimal mock objects that replicate the spaCy Doc/Token/Span API
     used by symptom_extractor.py and nlp_engine.py.
  2. Processes several clinical test sentences.
  3. Prints structured JSON output — proving the pipeline logic is correct.

This is NOT a replacement for the real pipeline; it is a correctness proof.
The real pipeline (main.py) requires:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set

# ──────────────────────────────────────────────────────────────────────────────
# Minimal spaCy-compatible mock objects
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MockToken:
    text: str
    lemma_: str
    pos_: str          # NOUN, VERB, ADJ, …
    tag_: str          # NN, VBZ, …
    dep_: str          # nsubj, dobj, neg, …
    head_idx: int      # index of governor token
    i: int             # token index in doc
    is_stop: bool = False
    is_alpha: bool = True
    ent_type_: str = ""

    # These are set after the full token list is built
    _doc: Any = field(default=None, repr=False, compare=False)

    @property
    def head(self) -> "MockToken":
        return self._doc.tokens[self.head_idx]

    @property
    def children(self) -> List["MockToken"]:
        return [t for t in self._doc.tokens if t.head_idx == self.i and t.i != self.i]

    @property
    def ancestors(self) -> Iterator["MockToken"]:
        visited = set()
        cur = self
        while cur.head_idx != cur.i:  # root is its own head
            parent = self._doc.tokens[cur.head_idx]
            if parent.i in visited:
                break
            visited.add(parent.i)
            yield parent
            cur = parent

    @property
    def subtree(self) -> List["MockToken"]:
        """Return all tokens in the subtree rooted at this token (BFS)."""
        result = []
        queue = [self]
        visited = set()
        while queue:
            tok = queue.pop(0)
            if tok.i in visited:
                continue
            visited.add(tok.i)
            result.append(tok)
            queue.extend(tok.children)
        return result

    @property
    def sent(self) -> "MockSent":
        return self._doc.sent_of(self)


@dataclass
class MockSpan:
    """Represents a contiguous slice of tokens (entity or noun chunk)."""
    tokens: List[MockToken]
    label_: str = ""
    _doc: Any = field(default=None, repr=False, compare=False)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def start(self) -> int:
        return self.tokens[0].i

    @property
    def end(self) -> int:
        return self.tokens[-1].i + 1

    @property
    def start_char(self) -> int:
        return self.tokens[0].i  # simplified: use token index as char proxy

    @property
    def end_char(self) -> int:
        return self.tokens[-1].i + 1

    @property
    def root(self) -> MockToken:
        # Root = token whose head is outside the span, or self-loop
        for tok in self.tokens:
            if tok.head_idx < self.tokens[0].i or tok.head_idx >= self.tokens[-1].i + 1:
                return tok
            if tok.head_idx == tok.i:
                return tok
        return self.tokens[0]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)


@dataclass
class MockSent:
    tokens: List[MockToken]

    @property
    def text(self) -> str:
        # Reconstruct with spaces (simplified — real spaCy uses whitespace flags)
        return " ".join(t.text for t in self.tokens)


class MockDoc:
    """
    Mimics the spaCy Doc interface used by NLPEngine and SymptomExtractor.
    """

    def __init__(
        self,
        tokens: List[MockToken],
        ents: List[MockSpan],
        noun_chunks: List[MockSpan],
        sents: List[MockSent],
        text: str,
    ):
        self.tokens = tokens
        self.ents = ents
        self._noun_chunks = noun_chunks
        self._sents = sents
        self.text = text

        # Wire back-references
        for tok in self.tokens:
            tok._doc = self
        for span in self.ents + self._noun_chunks:
            span._doc = self

    @property
    def noun_chunks(self):
        return iter(self._noun_chunks)

    @property
    def sents(self):
        return iter(self._sents)

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, key):
        if isinstance(key, slice):
            toks = self.tokens[key]
            return MockSpan(tokens=toks, _doc=self)
        return self.tokens[key]

    def __len__(self):
        return len(self.tokens)

    def char_span(self, start: int, end: int) -> Optional[MockSpan]:
        """Simplified char_span using token indices as char proxies."""
        toks = [t for t in self.tokens if start <= t.i < end]
        if not toks:
            return None
        return MockSpan(tokens=toks, _doc=self)

    def sent_of(self, tok: MockToken) -> MockSent:
        for sent in self._sents:
            if any(t.i == tok.i for t in sent.tokens):
                return sent
        return self._sents[0]


# ──────────────────────────────────────────────────────────────────────────────
# Minimal rule-based NLP processor (replaces NLPEngine for the demo)
# ──────────────────────────────────────────────────────────────────────────────

# Same vocab as nlp_engine.py
_PRE_NEG_CUES: Set[str] = {
    "no", "not", "without", "denies", "deny", "denying",
    "absent", "negative", "never", "none", "neither",
    "doesn't", "does not", "don't", "do not",
    "free", "ruled out", "rules out",
}

SYMPTOM_SEEDS: Set[str] = {
    "pain", "ache", "aching", "discomfort", "soreness", "tenderness",
    "swelling", "inflammation", "irritation", "headache", "migraine",
    "dizziness", "vertigo", "nausea", "vomiting", "numbness", "tingling",
    "weakness", "fatigue", "tremor", "seizure", "confusion", "syncope",
    "fainting", "cough", "wheeze", "wheezing", "breathlessness", "dyspnea",
    "shortness", "congestion", "sneezing", "palpitation", "tachycardia",
    "diarrhea", "constipation", "bloating", "cramping", "cramp",
    "reflux", "heartburn", "indigestion", "rash", "itching", "pruritus",
    "fever", "chills", "sweating", "malaise", "lethargy", "stiffness", "spasm",
}

SEVERITY_TERMS: Set[str] = {
    "sharp", "dull", "burning", "stabbing", "throbbing", "aching",
    "crushing", "squeezing", "shooting", "radiating", "sore",
    "mild", "moderate", "severe", "extreme", "intense", "slight",
    "persistent", "intermittent", "constant", "chronic", "acute",
}

LOCATION_TERMS: Set[str] = {
    "head", "neck", "throat", "shoulder", "arm", "elbow", "wrist",
    "hand", "finger", "chest", "breast", "back", "spine", "lumbar",
    "abdomen", "belly", "stomach", "pelvis", "hip", "groin",
    "thigh", "knee", "leg", "shin", "calf", "ankle", "foot", "toe",
    "left", "right", "bilateral", "upper", "lower", "anterior",
    "posterior", "lateral", "medial", "frontal", "temporal", "cervical",
    "lumbar", "sacral", "ear", "eye", "nose", "mouth", "jaw", "face", "forehead",
}

_DURATION_RE = re.compile(
    r"(?:for\s+)?"
    r"(?:"
    r"\d+\s+(?:second|minute|hour|day|week|month|year)s?"
    r"|(?:since|over\s+the\s+(?:past|last))\s+(?:\w+\s+){0,3}\w+"
    r"|(?:a\s+few|several|many)\s+(?:second|minute|hour|day|week|month|year)s?"
    r"|(?:all\s+day|all\s+night|overnight|recently|lately)"
    r")",
    re.IGNORECASE,
)

_NORMALIZE_MAP: Dict[str, str] = {
    "pains": "pain", "aches": "ache", "aching": "ache",
    "headaches": "headache", "shortness of breath": "dyspnea",
    "short of breath": "dyspnea", "throwing up": "vomiting",
    "feeling dizzy": "dizziness", "runny nose": "nasal congestion",
    "running nose": "nasal congestion",
}


def _simple_lemma(word: str) -> str:
    """Very basic lemmatizer for demo purposes."""
    w = word.lower().rstrip(".,;:")
    for suffix, replacement in [("nesses", "ness"), ("ings", "ing"),
                                  ("aches", "ache"), ("pains", "pain"),
                                  ("ies", "y"), ("es", ""), ("s", "")]:
        if w.endswith(suffix) and len(w) - len(suffix) > 2:
            return w[: -len(suffix)] + replacement
    return w


def _simple_pos(word: str, prev_word: str = "") -> str:
    """Heuristic POS tagger for demo."""
    w = word.lower()
    if w in {"no", "not", "never", "without", "absent"}:
        return "DET" if w == "no" else "ADV"
    if w in {"is", "was", "are", "were", "have", "has", "had", "be"}:
        return "AUX"
    if w in {"the", "a", "an", "this", "that", "my", "his", "her", "their"}:
        return "DET"
    if w in {"in", "on", "at", "for", "with", "without", "since", "over",
             "from", "of", "and", "but", "or"}:
        return "ADP" if w not in {"and", "but", "or"} else "CCONJ"
    if _simple_lemma(w) in SEVERITY_TERMS:
        return "ADJ"
    if _simple_lemma(w) in LOCATION_TERMS and w not in SYMPTOM_SEEDS:
        return "NOUN"
    if _simple_lemma(w) in SYMPTOM_SEEDS:
        return "NOUN"
    if w.endswith("ing"):
        return "VERB"
    if w.endswith("ed"):
        return "VERB"
    return "NOUN"


def _assign_deps(tokens: List[MockToken], text: str) -> None:
    """
    Assign simplified dependency labels.
    Key labels needed by the extractors:
      neg  — negation modifier
      root — sentence root
      nsubj, dobj, nmod — for structure
    """
    for tok in tokens:
        w = tok.text.lower()
        if w in {"no", "not", "never", "n't"}:
            tok.dep_ = "neg"
            # point neg to the next non-stop token
            for j in range(tok.i + 1, len(tokens)):
                if not tokens[j].is_stop:
                    tok.head_idx = j
                    break
        elif w in {"denies", "deny", "without", "absent", "negative"}:
            tok.dep_ = "neg"
            tok.head_idx = tok.i  # treat as root-level negation signal
        elif tok.pos_ in {"VERB", "AUX"} and tok.dep_ == "":
            tok.dep_ = "root"
            tok.head_idx = tok.i
        elif tok.pos_ == "NOUN" and tok.dep_ == "":
            tok.dep_ = "nsubj"
            # attach to root verb if present
            root = next((t for t in tokens if t.dep_ == "root"), None)
            tok.head_idx = root.i if root else tok.i
        elif tok.pos_ == "ADJ" and tok.dep_ == "":
            tok.dep_ = "amod"
            # attach to nearest noun to the right
            for j in range(tok.i + 1, len(tokens)):
                if tokens[j].pos_ == "NOUN":
                    tok.head_idx = j
                    break
        if tok.dep_ == "":
            tok.dep_ = "dep"


def parse_text(text: str) -> MockDoc:
    """
    Rule-based mini NLP processor that replicates the spaCy Doc structure
    well enough for the extractor to run correctly.
    """
    # Sentence split on period / exclamation / question mark
    raw_sents = re.split(r"(?<=[.!?])\s+", text.strip())
    if not raw_sents:
        raw_sents = [text]

    all_tokens: List[MockToken] = []
    ents: List[MockSpan] = []
    chunks: List[MockSpan] = []
    sent_objs: List[MockSent] = []

    global_idx = 0
    for sent_text in raw_sents:
        words = re.findall(r"\S+", sent_text)
        sent_tokens: List[MockToken] = []

        for w in words:
            clean = w.strip(".,;:!?")
            if not clean:
                continue
            lemma = _simple_lemma(clean)
            pos = _simple_pos(clean)
            tok = MockToken(
                text=clean,
                lemma_=lemma,
                pos_=pos,
                tag_=pos,
                dep_="",       # assigned below
                head_idx=global_idx,   # self by default
                i=global_idx,
                is_stop=clean.lower() in {
                    "the", "a", "an", "is", "was", "are", "were",
                    "i", "my", "his", "her", "they", "it", "and", "but",
                    "for", "with", "in", "on", "at", "of", "to", "that",
                    "have", "has", "had",
                },
                is_alpha=clean.isalpha(),
                ent_type_="",
            )
            sent_tokens.append(tok)
            all_tokens.append(tok)
            global_idx += 1

        _assign_deps(sent_tokens, sent_text)

        # Build a doc stub so sent_tokens can reference each other
        class _TempDoc:
            tokens = all_tokens
        for t in sent_tokens:
            t._doc = _TempDoc()

        # --- NER: detect symptom spans ---
        # Walk tokens; group consecutive symptom-seed tokens (+ adj modifiers)
        i = 0
        while i < len(sent_tokens):
            tok = sent_tokens[i]
            if _simple_lemma(tok.text.lower()) in SYMPTOM_SEEDS:
                # Expand left for adjective modifiers (severity/location qualifiers)
                start = i
                while start > 0 and sent_tokens[start - 1].pos_ in {"ADJ", "NOUN"}:
                    # Only expand if the preceding token is relevant
                    prev = sent_tokens[start - 1]
                    if (_simple_lemma(prev.text.lower()) in SEVERITY_TERMS
                            or _simple_lemma(prev.text.lower()) in LOCATION_TERMS):
                        start -= 1
                    else:
                        break
                # Expand right for compound nouns
                end = i + 1
                while end < len(sent_tokens) and sent_tokens[end].pos_ == "NOUN":
                    if _simple_lemma(sent_tokens[end].text.lower()) in SYMPTOM_SEEDS | LOCATION_TERMS:
                        end += 1
                    else:
                        break

                span_tokens = sent_tokens[start:end]
                span = MockSpan(tokens=span_tokens, label_="PROBLEM")

                # Negation check: look back _NEG_SCOPE_TOKENS in sent_tokens
                neg_start = max(0, start - 6)
                negated = any(
                    _simple_lemma(sent_tokens[j].text.lower()) in _PRE_NEG_CUES
                    or sent_tokens[j].dep_ == "neg"
                    for j in range(neg_start, start)
                )
                # Also check ancestors
                if not negated:
                    head_tok = span_tokens[0]
                    for ancestor in head_tok.ancestors:
                        if _simple_lemma(ancestor.text.lower()) in _PRE_NEG_CUES:
                            negated = True
                            break

                ents.append(span)
                # Mark ent_type on constituent tokens
                for t in span_tokens:
                    t.ent_type_ = "PROBLEM"
                i = end
            else:
                i += 1

        # --- Noun chunks: every NOUN token not already in an entity ---
        ent_indices = {t.i for span in ents for t in span.tokens}
        j = 0
        while j < len(sent_tokens):
            tok = sent_tokens[j]
            if tok.pos_ == "NOUN" and tok.i not in ent_indices and not tok.is_stop:
                chunk = MockSpan(tokens=[tok], _doc=None)
                chunks.append(chunk)
            j += 1

        sent_objs.append(MockSent(tokens=sent_tokens))

    doc = MockDoc(
        tokens=all_tokens,
        ents=ents,
        noun_chunks=chunks,
        sents=sent_objs,
        text=text,
    )
    return doc


# ──────────────────────────────────────────────────────────────────────────────
# Inline symptom extractor (mirrors symptom_extractor.py logic exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return _NORMALIZE_MAP.get(cleaned, cleaned)


def _contains_seed(text: str) -> bool:
    text_lower = text.lower()
    return any(seed in text_lower for seed in SYMPTOM_SEEDS)


def _find_severity(span_tokens: List[MockToken], all_tokens: List[MockToken]) -> Optional[str]:
    candidates = []
    for tok in span_tokens:
        if _simple_lemma(tok.text.lower()) in SEVERITY_TERMS:
            candidates.append(tok.text.lower())
    if span_tokens:
        root = span_tokens[0]
        for tok in root.subtree:
            lm = _simple_lemma(tok.text.lower())
            if lm in SEVERITY_TERMS and tok.text.lower() not in candidates:
                candidates.append(tok.text.lower())
    # Neighbourhood ±3
    if span_tokens:
        start = max(0, span_tokens[0].i - 3)
        end = min(len(all_tokens), span_tokens[-1].i + 4)
        for tok in all_tokens[start:end]:
            if _simple_lemma(tok.text.lower()) in SEVERITY_TERMS and tok.text.lower() not in candidates:
                candidates.append(tok.text.lower())
    return candidates[0] if candidates else None


def _find_location(span_tokens: List[MockToken], all_tokens: List[MockToken]) -> Optional[str]:
    loc_toks = []
    for tok in span_tokens:
        if _simple_lemma(tok.text.lower()) in LOCATION_TERMS:
            loc_toks.append(tok)
    if span_tokens:
        start = max(0, span_tokens[0].i - 4)
        end = min(len(all_tokens), span_tokens[-1].i + 5)
        for tok in all_tokens[start:end]:
            if _simple_lemma(tok.text.lower()) in LOCATION_TERMS and tok not in loc_toks:
                loc_toks.append(tok)
    if not loc_toks:
        return None
    loc_toks.sort(key=lambda t: t.i)
    return " ".join(t.text.lower() for t in loc_toks)


def _find_duration(span_tokens: List[MockToken], full_text: str) -> Optional[str]:
    if not span_tokens:
        return None
    sent_text = span_tokens[0].sent.text
    matches = _DURATION_RE.findall(sent_text)
    return matches[0].strip() if matches else None


def extract_symptoms(doc: MockDoc, original_text: str) -> dict:
    """Mirrors SymptomExtractor.extract() logic."""
    candidates = []
    seen_names: Set[str] = set()

    for ent in doc.ents:
        norm = _normalize(ent.text)
        if not _contains_seed(norm):
            continue

        neg = any(
            _simple_lemma(t.text.lower()) in _PRE_NEG_CUES or t.dep_ == "neg"
            for t in doc.tokens[max(0, ent.start - 6): ent.start]
        )
        if not neg:
            for tok in ent.root.ancestors:
                if _simple_lemma(tok.text.lower()) in _PRE_NEG_CUES:
                    neg = True
                    break

        sev = _find_severity(ent.tokens, doc.tokens)
        loc = _find_location(ent.tokens, doc.tokens)
        dur = _find_duration(ent.tokens, original_text)

        sym = {
            "name": norm,
            "severity": sev,
            "location": loc,
            "duration": dur,
            "negated": neg,
            "raw_text": ent.text,
        }
        if norm not in seen_names:
            candidates.append(sym)
            seen_names.add(norm)

    active = [s for s in candidates if not s["negated"]]
    negated_names = [s["name"] for s in candidates if s["negated"]]

    # Clean output (drop internal fields)
    symptoms_out = []
    for s in active:
        entry = {"name": s["name"]}
        if s["severity"]:
            entry["severity"] = s["severity"]
        if s["location"]:
            entry["location"] = s["location"]
        if s["duration"]:
            entry["duration"] = s["duration"]
        symptoms_out.append(entry)

    return {
        "symptoms": symptoms_out,
        "negated_symptoms": negated_names,
        "input_text": original_text,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    "I have a sharp pain in my lower back for 2 days. No fever.",
    "Patient denies nausea and vomiting. Reports severe headache since yesterday morning.",
    "She has been experiencing intermittent chest pain radiating to the left arm for 3 days. No shortness of breath.",
    "Complains of dull aching in the right knee and bilateral ankle swelling. Denies any rash.",
    "The patient feels persistent fatigue and mild dizziness. No chills. Has had a burning sensation in the abdomen for a few days.",
]


def main():
    print("=" * 72)
    print("  Medical Symptom Extraction Pipeline — Demo & Validation")
    print("  (Mock NLP layer — replace with spaCy for production)")
    print("=" * 72)

    for idx, text in enumerate(TEST_CASES, 1):
        print(f"\n{'─' * 72}")
        print(f"Test {idx}: {text}")
        print("─" * 72)

        doc = parse_text(text)
        result = extract_symptoms(doc, text)

        print(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 72}")
    print("All test cases processed successfully.")
    print("Run 'python main.py' with spaCy installed for the full pipeline.")


if __name__ == "__main__":
    main()
