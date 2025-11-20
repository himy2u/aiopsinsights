"""PII redaction utilities for healthcare data."""
import re
from typing import Dict, List, Optional, Pattern, Set

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.nlp_engine import NlpEngine

# Custom patterns for healthcare-specific PII
HEALTHCARE_PATTERNS = [
    {
        "name": "MEDICAL_RECORD_NUMBER",
        "regex": r"\b\d{4,}-\d{4,}\b",
        "score": 0.9,
    },
    {
        "name": "HEALTH_INSURANCE_NUMBER",
        "regex": r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN-like
        "score": 0.9,
    },
    {
        "name": "PHONE_NUMBER",
        "regex": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "score": 0.8,
    },
]

# Common PII patterns
COMMON_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "SSN": r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",
}


class PIIDetector:
    """Detect and redact PII from text using regex and NLP."""

    def __init__(self, use_ml: bool = False):
        """Initialize the PII detector.

        Args:
            use_ml: Whether to use ML-based NER (slower but more accurate)
        """
        self.use_ml = use_ml
        self.patterns = self._compile_patterns()
        self.analyzer = self._init_analyzer() if use_ml else None

    def _compile_patterns(self) -> Dict[str, Pattern]:
        """Compile regex patterns for PII detection."""
        return {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in COMMON_PATTERNS.items()
        }

    def _init_analyzer(self) -> AnalyzerEngine:
        """Initialize the Presidio analyzer with custom patterns."""
        # Use small spaCy model for faster inference
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }

        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # Add custom recognizers
        custom_recognizers = [
            PatternRecognizer(
                supported_entity=pattern["name"],
                patterns=[Pattern(pattern["regex"], pattern["score"])],
            )
            for pattern in HEALTHCARE_PATTERNS
        ]

        return AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en"],
            registry=custom_recognizers,
        )

    def detect_pii(self, text: str) -> Set[str]:
        """Detect PII in the given text.

        Args:
            text: Input text to scan for PII

        Returns:
            Set of PII types found in the text
        """
        pii_types = set()

        # Check regex patterns
        for pii_type, pattern in self.patterns.items():
            if pattern.search(text):
                pii_types.add(pii_type)

        # Use ML-based detection if enabled
        if self.use_ml and self.analyzer:
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=["PERSON", "LOCATION", "DATE_TIME"],
                score_threshold=0.7,
            )
            for result in results:
                pii_types.add(result.entity_type)

        return pii_types

    def redact_text(
        self, text: str, replacement: str = "[REDACTED]", pii_types: Optional[Set[str]] = None
    ) -> str:
        """Redact PII from the given text.

        Args:
            text: Input text to redact
            replacement: String to replace PII with
            pii_types: Specific PII types to redact (if None, redact all)

        Returns:
            Text with PII redacted
        """
        if not text:
            return text

        if pii_types is None:
            pii_types = set(COMMON_PATTERNS.keys()) | {
                pattern["name"] for pattern in HEALTHCARE_PATTERNS
            }

        # Redact regex patterns
        for pii_type, pattern in self.patterns.items():
            if pii_types and pii_type not in pii_types:
                continue
            text = pattern.sub(replacement, text)

        # Redact ML-detected entities if enabled
        if self.use_ml and self.analyzer:
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=["PERSON", "LOCATION", "DATE_TIME"],
                score_threshold=0.7,
            )
            # Sort by start position in reverse to avoid offset issues
            for result in sorted(results, key=lambda x: x.start, reverse=True):
                if pii_types and result.entity_type not in pii_types:
                    continue
                text = text[: result.start] + replacement + text[result.end :]

        return text


# Singleton instance with ML disabled by default for performance
pii_detector = PIIDetector(use_ml=False)
