"""
nlp_engine.py
-------------
Loads the NLP model, attaches negation detection, and exposes a single
`process(text)` method that returns a rich linguistic feature dict.

Design decision:
  We use spaCy's `en_core_web_sm` as the default model because it ships
  with a full dependency parser and POS tagger out of the box and needs
  no extra downloads beyond the model itself.

  If SciSpacy is available (pip install scispacy + model), swap the model
  name to "en_core_sci_sm" for biomedical NER trained on MIMIC/MedMentions.

Negation:
  We implement a rule-based negation scope detector instead of negspacy
  so there are zero additional install requirements. The approach follows
  the NegEx algorithm family: scan a fixed-token window BEFORE each entity
  for negation cues (no, denies, without, not, …).  This is medically
  grounded and interpretable.
"""

from __future__ import annotations
from typing import Any, Dict, List

import spacy
from spacy.tokens import Doc, Span

# ---------------------------------------------------------------------------
# Negation cue vocabulary (pre-negation triggers)
# These fire when they appear BEFORE a symptom within a dependency window.
# ---------------------------------------------------------------------------
_PRE_NEG_CUES: set[str] = {
    "no", "not", "without", "denies", "deny", "denying",
    "absent", "negative", "never", "none", "neither",
    "doesn't", "does not", "don't", "do not",
    "isn't", "is not", "wasn't", "was not",
    "didn't", "did not", "haven't", "has not",
    "free", "ruled out", "rules out", "unremarkable",
}

# How many tokens to look back from the entity head for a negation cue.
_NEG_SCOPE_TOKENS = 6


class NLPEngine:
    """
    Wraps a spaCy pipeline with negation-detection capability.

    Usage
    -----
    engine = NLPEngine()
    features = engine.process("Patient denies fever but reports sharp chest pain.")
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """
        Load the spaCy model.  The pipeline needs:
          - tok2vec / tagger  → POS tags
          - parser            → dependency arcs (for negation scope)
          - ner               → entity spans
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> Dict[str, Any]:
        """
        Run the full NLP pipeline on *text* and return a structured dict
        containing every linguistic layer needed by downstream extractors.

        Returns
        -------
        {
          "doc":        spaCy Doc object (for extractor re-use),
          "tokens":     list of token dicts,
          "entities":   list of entity dicts (with negation flag),
          "noun_chunks":list of noun-chunk dicts,
          "sentences":  list of sentence strings,
        }
        """
        doc: Doc = self.nlp(text)

        tokens = self._extract_tokens(doc)
        entities = self._extract_entities(doc)
        noun_chunks = self._extract_noun_chunks(doc)
        sentences = [sent.text.strip() for sent in doc.sents]

        return {
            "doc": doc,                   # raw spaCy Doc for extractor re-use
            "tokens": tokens,
            "entities": entities,
            "noun_chunks": noun_chunks,
            "sentences": sentences,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_tokens(self, doc: Doc) -> List[Dict[str, Any]]:
        """
        Return per-token linguistic annotations.

        POS and dependency tags are the raw spaCy strings so callers can
        apply their own filters without re-processing.
        """
        return [
            {
                "text": tok.text,
                "lemma": tok.lemma_,
                "pos": tok.pos_,          # Universal POS (NOUN, VERB, ADJ…)
                "tag": tok.tag_,          # Fine-grained Penn tag (NN, VBZ…)
                "dep": tok.dep_,          # Dependency relation label
                "head": tok.head.text,    # Governor token
                "is_stop": tok.is_stop,
                "is_alpha": tok.is_alpha,
                "ent_type": tok.ent_type_ or None,
            }
            for tok in doc
        ]

    def _extract_entities(self, doc: Doc) -> List[Dict[str, Any]]:
        """
        Return NER spans enriched with a negation flag.

        Negation detection strategy
        ---------------------------
        For each entity span we look at the `head` token of the span
        (the syntactic root).  We then walk the left siblings and the
        left context window of *_NEG_SCOPE_TOKENS* tokens for any
        negation cue lemma.  This covers:
          - "no fever"              (direct modifier)
          - "patient denies fever"  (verb-mediated negation)
          - "chest pain absent"     (post-nominal predicate – caught via
                                    the sentence-level scan)
        """
        entities = []
        for ent in doc.ents:
            negated = self._is_negated(ent, doc)
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "negated": negated,
                }
            )
        return entities

    def _extract_noun_chunks(self, doc: Doc) -> List[Dict[str, Any]]:
        """
        Noun chunks give us candidate symptom phrases even when the NER
        model doesn't fire (e.g. for compound anatomical terms not in
        the training vocabulary).

        Each chunk is also tested for negation so the extractor can
        treat noun chunks and NER entities uniformly.
        """
        chunks = []
        for chunk in doc.noun_chunks:
            negated = self._is_negated(chunk, doc)
            chunks.append(
                {
                    "text": chunk.text,
                    "root_text": chunk.root.text,
                    "root_dep": chunk.root.dep_,
                    "root_pos": chunk.root.pos_,
                    "negated": negated,
                }
            )
        return chunks

    def _is_negated(self, span: Span, doc: Doc) -> bool:
        """
        Determine whether *span* falls within a negation scope.

        Two complementary passes:
        1. Window scan  – look at the _NEG_SCOPE_TOKENS tokens immediately
                          to the LEFT of the span start.
        2. Ancestor scan – walk the dependency tree ancestors of the span
                           head looking for negation-cue verbs (denies,
                           without, …).  This catches long-distance negation
                           like "The patient consistently denies any fever."
        """
        head_token = span.root

        # --- Pass 1: left-context window ---
        window_start = max(0, span.start - _NEG_SCOPE_TOKENS)
        left_context = doc[window_start : span.start]
        for tok in left_context:
            if tok.lemma_.lower() in _PRE_NEG_CUES:
                return True
            # spaCy marks syntactic negation with dep_ == "neg"
            if tok.dep_ == "neg" and tok.head.i >= window_start:
                return True

        # --- Pass 2: dependency-tree ancestors ---
        for ancestor in head_token.ancestors:
            if ancestor.lemma_.lower() in _PRE_NEG_CUES:
                return True
            # If the ancestor carries a "neg" child, the whole subtree is negated
            for child in ancestor.children:
                if child.dep_ == "neg":
                    return True

        return False
