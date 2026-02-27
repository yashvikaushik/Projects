"""
symptom_extractor.py
--------------------
Accepts the rich linguistic feature dict produced by NLPEngine.process()
and returns an ExtractionResult containing structured symptom records.

Architecture notes
------------------
The extractor deliberately avoids hardcoded disease names.  It operates
on controlled *symptom-category* vocabulary lists – terms that indicate
a symptom concept rather than a diagnosis.  This keeps the module focused
on the "Input → NLP → Symptom Extraction" slice of the pipeline.

Three candidate sources feed the extractor (in priority order):
  1. NER entities  – highest signal; already classified by spaCy/SciSpacy.
  2. Noun chunks   – catches compound symptom phrases the NER model missed.
  3. Pattern sweep – dependency-subtree reconstruction for adjectival symptoms
                     (e.g. "feeling dizzy") that surface as VERB/ADJ, not NOUN.

Normalization converts surface forms to canonical terms via a small lookup
table and lemmatization so "pains" → "pain", "aching" → "ache", etc.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from spacy.tokens import Doc, Token

from schemas import ExtractedSymptom, ExtractionResult

# ---------------------------------------------------------------------------
# Controlled vocabulary – symptom seed terms
# ---------------------------------------------------------------------------
# These lemma forms anchor detection.  NOT an exhaustive list – the NLP
# pipeline handles unseen terms via NER + noun-chunk composition.
SYMPTOM_SEEDS: Set[str] = {
    # general
    "pain", "ache", "aching", "discomfort", "soreness", "tenderness",
    "swelling", "inflammation", "irritation",
    # head / neuro
    "headache", "migraine", "dizziness", "vertigo", "nausea", "vomiting",
    "numbness", "tingling", "weakness", "fatigue", "tremor", "seizure",
    "confusion", "syncope", "fainting",
    # respiratory
    "cough", "wheeze", "wheezing", "breathlessness", "dyspnea",
    "shortness", "congestion", "sneezing",
    # cardiac / circulatory
    "palpitation", "tachycardia", "bradycardia",
    # GI
    "diarrhea", "constipation", "bloating", "cramping", "cramp",
    "reflux", "heartburn", "indigestion",
    # skin
    "rash", "itching", "pruritus", "bruising", "lesion", "blister",
    # systemic
    "fever", "chills", "sweating", "malaise", "lethargy",
    # musculoskeletal
    "stiffness", "spasm",
    # eyes / ENT
    "blurring", "blurriness", "tinnitus", "hearing",
    # urinary
    "dysuria", "frequency", "urgency",
}

# ---------------------------------------------------------------------------
# Severity / intensity qualifiers
# ---------------------------------------------------------------------------
SEVERITY_TERMS: Set[str] = {
    "sharp", "dull", "burning", "stabbing", "throbbing", "aching",
    "crushing", "squeezing", "shooting", "radiating", "sore",
    "mild", "moderate", "severe", "extreme", "intense", "slight",
    "persistent", "intermittent", "constant", "chronic", "acute",
}

# ---------------------------------------------------------------------------
# Anatomical location terms
# ---------------------------------------------------------------------------
LOCATION_TERMS: Set[str] = {
    "head", "neck", "throat", "shoulder", "arm", "elbow", "wrist",
    "hand", "finger", "chest", "breast", "back", "spine", "lumbar",
    "abdomen", "belly", "stomach", "pelvis", "hip", "groin",
    "thigh", "knee", "leg", "shin", "calf", "ankle", "foot", "toe",
    "left", "right", "bilateral", "upper", "lower", "anterior",
    "posterior", "lateral", "medial", "frontal", "occipital",
    "temporal", "thoracic", "cervical", "lumbar", "sacral",
    "ear", "eye", "nose", "mouth", "jaw", "face", "forehead",
}

# ---------------------------------------------------------------------------
# Duration / temporal pattern words (for phrase anchoring)
# ---------------------------------------------------------------------------
_DURATION_ANCHORS: re.Pattern = re.compile(
    r"""
    (?:for\s+)?               # optional "for"
    (?:
        \d+\s+                # numeric quantity  e.g. "2 "
        (?:second|minute|hour|day|week|month|year)s?  # unit
        |
        (?:since|over\s+the\s+(?:past|last))\s+       # relational
        (?:\w+\s+){0,3}\w+    # e.g. "last week", "past few days"
        |
        (?:a\s+few|several|many)\s+
        (?:second|minute|hour|day|week|month|year)s?
        |
        (?:all\s+day|all\s+night|overnight|recently|lately)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Surface-form → canonical name normalization table
# ---------------------------------------------------------------------------
_NORMALIZE: Dict[str, str] = {
    "pains": "pain",
    "aches": "ache",
    "aching": "ache",
    "headaches": "headache",
    "migraines": "migraine",
    "shortness of breath": "dyspnea",
    "short of breath": "dyspnea",
    "sob": "dyspnea",
    "throwing up": "vomiting",
    "threw up": "vomiting",
    "passed out": "syncope",
    "blacking out": "syncope",
    "racing heart": "palpitation",
    "heart racing": "palpitation",
    "feeling dizzy": "dizziness",
    "feels dizzy": "dizziness",
    "feel dizzy": "dizziness",
    "running nose": "nasal congestion",
    "runny nose": "nasal congestion",
}


class SymptomExtractor:
    """
    Transforms NLPEngine output into an ExtractionResult.

    Usage
    -----
    extractor = SymptomExtractor()
    result = extractor.extract(nlp_features, original_text)
    """

    def extract(
        self,
        nlp_features: Dict[str, Any],
        original_text: str,
    ) -> ExtractionResult:
        """
        Main entry point.

        Parameters
        ----------
        nlp_features : dict returned by NLPEngine.process()
        original_text : the raw string the user submitted

        Returns
        -------
        ExtractionResult with .symptoms and .negated_symptoms populated.
        """
        doc: Doc = nlp_features["doc"]
        noun_chunks = nlp_features["noun_chunks"]
        entities = nlp_features["entities"]

        candidates: List[ExtractedSymptom] = []

        # --- Source 1: NER entities ---
        for ent in entities:
            sym = self._entity_to_symptom(ent, doc)
            if sym:
                candidates.append(sym)

        # --- Source 2: Noun chunks (de-duplicated against NER) ---
        ner_spans = {e["text"].lower() for e in entities}
        for chunk in noun_chunks:
            if chunk["text"].lower() in ner_spans:
                continue  # already captured by NER
            sym = self._chunk_to_symptom(chunk, doc)
            if sym:
                candidates.append(sym)

        # --- Source 3: Dependency-pattern sweep for verb/adj symptoms ---
        dep_syms = self._dep_sweep(doc, ner_spans)
        candidates.extend(dep_syms)

        # --- De-duplicate by normalized name ---
        candidates = self._deduplicate(candidates)

        # --- Split into active vs negated ---
        active = [s for s in candidates if not s.negated]
        negated_names = [s.name for s in candidates if s.negated]

        return ExtractionResult(
            symptoms=active,
            negated_symptoms=negated_names,
            input_text=original_text,
        )

    # ------------------------------------------------------------------
    # Source 1 – NER entity → symptom
    # ------------------------------------------------------------------

    def _entity_to_symptom(
        self,
        ent: Dict[str, Any],
        doc: Doc,
    ) -> Optional[ExtractedSymptom]:
        """
        Convert a spaCy NER entity dict to ExtractedSymptom if it looks
        like a symptom.

        en_core_web_sm labels symptoms under various categories.  We accept
        any entity whose text overlaps with the symptom-seed vocabulary OR
        whose label suggests a medical concept.  For SciSpacy models the
        label would be "PROBLEM" – we handle both.
        """
        text: str = ent["text"]
        norm = self._normalize(text)

        # Accept if seed match OR medical NER label
        is_medical_label = ent["label"] in {
            "PROBLEM", "DISEASE", "SYMPTOM",
            "CONDITION",  # custom models
        }
        is_seed_match = self._contains_seed(norm)

        if not (is_medical_label or is_seed_match):
            return None

        span_tokens = doc.char_span(ent["start"], ent["end"])
        if span_tokens is None:
            return None

        severity = self._find_severity(span_tokens, doc)
        location = self._find_location(span_tokens, doc)
        duration = self._find_duration(span_tokens, doc)

        return ExtractedSymptom(
            name=norm,
            severity=severity,
            location=location,
            duration=duration,
            negated=ent["negated"],
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Source 2 – Noun chunk → symptom
    # ------------------------------------------------------------------

    def _chunk_to_symptom(
        self,
        chunk: Dict[str, Any],
        doc: Doc,
    ) -> Optional[ExtractedSymptom]:
        """
        Evaluate a noun chunk as a candidate symptom.

        We filter aggressively: the root lemma or any constituent token
        must appear in SYMPTOM_SEEDS.
        """
        text: str = chunk["text"]
        norm = self._normalize(text)

        if not self._contains_seed(norm):
            return None

        # Retrieve the actual spaCy Span for contextual lookups
        span = self._find_span(doc, text)
        severity = self._find_severity(span, doc) if span else None
        location = self._find_location(span, doc) if span else None
        duration = self._find_duration(span, doc) if span else None

        return ExtractedSymptom(
            name=norm,
            severity=severity,
            location=location,
            duration=duration,
            negated=chunk["negated"],
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Source 3 – Dependency sweep for verb/adj symptoms
    # ------------------------------------------------------------------

    def _dep_sweep(
        self,
        doc: Doc,
        already_seen: Set[str],
    ) -> List[ExtractedSymptom]:
        """
        Catches symptom expressions that surface as verbs or adjectives
        rather than nouns, e.g.:
          - "I feel dizzy"     → "dizziness"
          - "She was vomiting" → "vomiting"

        Strategy: iterate all tokens; if the lemma is a symptom seed and
        the token's POS is VERB/AUX/ADJ and it wasn't captured by NER
        or noun-chunk passes, synthesize a symptom record.
        """
        results: List[ExtractedSymptom] = []

        for tok in doc:
            if tok.pos_ not in {"VERB", "AUX", "ADJ"}:
                continue
            lemma = tok.lemma_.lower()
            if lemma not in SYMPTOM_SEEDS:
                continue
            if tok.text.lower() in already_seen:
                continue

            norm = self._normalize(tok.text)
            negated = self._token_is_negated(tok)
            # Build a pseudo-span around this single token for context lookup
            span = doc[tok.i : tok.i + 1]
            severity = self._find_severity(span, doc)
            location = self._find_location(span, doc)
            duration = self._find_duration(span, doc)

            results.append(
                ExtractedSymptom(
                    name=norm,
                    severity=severity,
                    location=location,
                    duration=duration,
                    negated=negated,
                    raw_text=tok.text,
                )
            )
            already_seen.add(tok.text.lower())

        return results

    # ------------------------------------------------------------------
    # Contextual attribute finders
    # ------------------------------------------------------------------

    def _find_severity(self, span, doc: Doc) -> Optional[str]:
        """
        Look for severity/intensity qualifiers in:
          a) the span's own tokens (e.g. "sharp chest pain")
          b) tokens in the span's dependency subtree
          c) immediate left/right neighbours (±3 tokens)
        """
        if span is None:
            return None

        candidates: List[str] = []

        # Within span
        for tok in span:
            if tok.lemma_.lower() in SEVERITY_TERMS:
                candidates.append(tok.text.lower())

        # Subtree of span root
        for tok in span.root.subtree:
            if tok.lemma_.lower() in SEVERITY_TERMS and tok.text not in candidates:
                candidates.append(tok.text.lower())

        # Neighbourhood window ±3
        start = max(0, span.start - 3)
        end = min(len(doc), span.end + 3)
        for tok in doc[start:end]:
            if tok.lemma_.lower() in SEVERITY_TERMS and tok.text not in candidates:
                candidates.append(tok.text.lower())

        # Return the first (leftmost) qualifier found; multiple can be joined if needed
        return candidates[0] if candidates else None

    def _find_location(self, span, doc: Doc) -> Optional[str]:
        """
        Identify anatomical location by:
          a) scanning span tokens for location seeds
          b) scanning the dependency subtree
          c) looking at adjacent noun chunks that consist of location terms
        """
        if span is None:
            return None

        location_tokens: List[Token] = []

        # Subtree walk
        for tok in span.root.subtree:
            if tok.lemma_.lower() in LOCATION_TERMS:
                location_tokens.append(tok)

        # Neighbourhood ±4
        start = max(0, span.start - 4)
        end = min(len(doc), span.end + 4)
        for tok in doc[start:end]:
            if tok.lemma_.lower() in LOCATION_TERMS and tok not in location_tokens:
                location_tokens.append(tok)

        if not location_tokens:
            return None

        # Sort by position and join to reconstruct multi-word locations
        location_tokens.sort(key=lambda t: t.i)
        return " ".join(t.text.lower() for t in location_tokens)

    def _find_duration(self, span, doc: Doc) -> Optional[str]:
        """
        Extract duration/temporal phrases using regex on the sentence
        containing the symptom span.

        Regex-based duration extraction is more reliable than dependency
        parsing for free-form temporal expressions ("since last Tuesday",
        "for about 3 days").
        """
        if span is None:
            return None

        # Identify the containing sentence
        sent = span.root.sent
        sent_text = sent.text

        matches = _DURATION_ANCHORS.findall(sent_text)
        if matches:
            # Return the first (most salient) duration phrase, stripped
            return matches[0].strip()

        return None

    # ------------------------------------------------------------------
    # Token-level negation check (for dep_sweep)
    # ------------------------------------------------------------------

    @staticmethod
    def _token_is_negated(tok: Token) -> bool:
        """
        Check if a single token carries negation via its direct children
        or its ancestors.
        """
        # Direct "neg" child
        for child in tok.children:
            if child.dep_ == "neg":
                return True
        # Ancestor negation
        for ancestor in tok.ancestors:
            for child in ancestor.children:
                if child.dep_ == "neg":
                    return True
        return False

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Map surface text to a canonical symptom name.

        Steps:
          1. Lowercase and strip.
          2. Check explicit lookup table first (handles multi-word idioms).
          3. Return cleaned text (lemmatization is handled at the token
             level by spaCy; here we just tidy the string).
        """
        cleaned = text.strip().lower()
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned)
        return _NORMALIZE.get(cleaned, cleaned)

    @staticmethod
    def _contains_seed(text: str) -> bool:
          """Return True if any symptom seed appears as a substring in text."""
          text_lower = text.lower()
          return any(seed in text_lower for seed in SYMPTOM_SEEDS)

    #This part did some errors on time , "Couldn't identify WEEk in time field"
    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip().lower())
        # Strip leading articles that leak in from noun chunk boundaries
        cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned)
        return _NORMALIZE.get(cleaned, cleaned)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _find_span(doc: Doc, text: str):
        """Locate the first occurrence of *text* in *doc* as a Span."""
        text_lower = text.lower()
        doc_lower = doc.text.lower()
        idx = doc_lower.find(text_lower)
        if idx == -1:
            return None
        return doc.char_span(idx, idx + len(text))

    @staticmethod
    def _deduplicate(candidates: List[ExtractedSymptom]) -> List[ExtractedSymptom]:
        """
        Remove duplicate symptom records by normalized name.
        Prefer entries with more fields populated (location/duration/severity).
        """
        seen: Dict[str, ExtractedSymptom] = {}
        for sym in candidates:
            key = sym.name
            if key not in seen:
                seen[key] = sym
            else:
                # Score by field completeness
                existing = seen[key]
                existing_score = sum([
                    existing.severity is not None,
                    existing.location is not None,
                    existing.duration is not None,
                ])
                new_score = sum([
                    sym.severity is not None,
                    sym.location is not None,
                    sym.duration is not None,
                ])
                if new_score > existing_score:
                    seen[key] = sym
        return list(seen.values())
