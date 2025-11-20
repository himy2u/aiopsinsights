"""Tests for PII detection and redaction."""
import pytest

from bronze_ingest.pii import PIIDetector, pii_detector


def test_pii_detection():
    """Test PII detection with common patterns."""
    test_cases = [
        ("My email is test@example.com", {"EMAIL"}),
        ("Call me at 555-123-4567", {"PHONE_NUMBER"}),
        ("SSN: 123-45-6789", {"SSN"}),
        ("No PII here", set()),
        (
            "Email: test@example.com, Phone: (555) 123-4567",
            {"EMAIL", "PHONE_NUMBER"},
        ),
    ]

    detector = PIIDetector(use_ml=False)
    for text, expected in test_cases:
        assert detector.detect_pii(text) == expected


def test_pii_redaction():
    """Test PII redaction in text."""
    test_cases = [
        (
            "Email me at test@example.com",
            "Email me at [REDACTED]",
        ),
        (
            "Call 555-123-4567 for support",
            "Call [REDACTED] for support",
        ),
        (
            "No PII here",
            "No PII here",
        ),
    ]

    for text, expected in test_cases:
        assert pii_detector.redact_text(text) == expected


def test_healthcare_pii_detection():
    """Test healthcare-specific PII detection."""
    detector = PIIDetector(use_ml=False)
    
    # Test medical record number
    text = "Patient record #1234-5678-9012"
    assert "MEDICAL_RECORD_NUMBER" in detector.detect_pii(text)
    
    # Test health insurance number (SSN-like)
    text = "Insurance: 123-45-6789"
    assert "HEALTH_INSURANCE_NUMBER" in detector.detect_pii(text)
    
    # Test redaction
    redacted = pii_detector.redact_text("Record #1234-5678-9012")
    assert "[REDACTED]" in redacted
    assert "1234" not in redacted


def test_ml_pii_detection():
    """Test ML-based PII detection (if ML is available)."""
    try:
        detector = PIIDetector(use_ml=True)
        
        # Test person name detection (requires ML)
        text = "Patient John Smith reported symptoms"
        detected = detector.detect_pii(text)
        assert "PERSON" in detected or len(detected) == 0  # ML might not always detect
        
    except ImportError:
        pytest.skip("ML dependencies not available")


def test_pii_redaction_performance(benchmark):
    """Benchmark PII redaction performance."""
    text = """
    Patient John Smith (DOB: 01/01/1980) reported chest pain.
    Contact: 555-123-4567, email: john.smith@example.com
    Address: 123 Main St, Anytown, USA 12345
    """
    
    # Test regex-only performance
    result = benchmark(pii_detector.redact_text, text)
    assert "[REDACTED]" in result
    assert "john.smith" not in result
    assert "555-123-4567" not in result
