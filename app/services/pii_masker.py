"""
PII Masking Service

Detects and redacts personally identifiable information (PII) from user queries
before they reach the LLM, cache, or logs. Supports Indian and Ethiopian PII formats.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Agricultural context words — if these appear near a 10-digit number,
# the number is likely a price/quantity, not a phone number.
_AGRI_CONTEXT_WORDS = re.compile(
    r'(?:price|quintal|kg|kilogram|ton|acre|hectare|ዋጋ|ኩንታል|ኪሎ|ብር|birr)',
    re.IGNORECASE,
)

# Banking context words — a 9-18 digit number is only treated as a bank
# account if one of these appears nearby.
_BANK_CONTEXT_WORDS = re.compile(
    r'(?:account|a/c|acct|bank|saving|current|ባንክ|አካውንት)',
    re.IGNORECASE,
)


class PIIMasker:
    """Singleton service that masks PII in text using compiled regex patterns."""

    def __init__(self):
        # --- Compiled patterns (order matters — checked sequentially) ---

        # 1. UPI ID  (must be checked BEFORE email)
        #    e.g. user@ybl, name@paytm, id@oksbi, id@upi
        self._upi_re = re.compile(
            r'[\w.\-]+@(?:ybl|paytm|oksbi|okaxis|okicici|okhdfcbank|upi|apl|ibl|axl|sbi|icici|hdfcbank)\b',
            re.IGNORECASE,
        )

        # 2. Email
        self._email_re = re.compile(
            r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        )

        # 3. Aadhaar  — strictly 4-4-4 with separators (space, dash, dot)
        self._aadhaar_re = re.compile(
            r'\b\d{4}[\s\-\.]\d{4}[\s\-\.]\d{4}\b',
        )

        # 4. PAN Card  — ABCDE1234F
        self._pan_re = re.compile(
            r'\b[A-Z]{5}\d{4}[A-Z]\b',
        )

        # 5. IFSC Code — 4 letters + 0 + 6 alphanum
        self._ifsc_re = re.compile(
            r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
        )

        # 6. Indian phone with +91 prefix
        self._phone_intl_in_re = re.compile(
            r'(?:\+91[\s\-]?)?(?:\d[\s\-]?){10}(?=\s|$|[,;.\)\]}>])',
        )

        # 7. Ethiopian phone  +251 XX XXX XXXX (9 digits after country code)
        self._phone_et_re = re.compile(
            r'\+251[\s\-]?\d[\s\-]?\d[\s\-]?\d{3}[\s\-]?\d{4}\b',
        )

        # 8. Indian phone — bare 10 digits starting with 6-9
        self._phone_bare_re = re.compile(
            r'\b[6-9]\d{9}\b',
        )

        # 9. Ethiopian National ID — 2-3 uppercase letters + 6-10 digits
        self._eth_natid_re = re.compile(
            r'\b[A-Z]{2,3}\d{6,10}\b',
        )

        # 10. Bank account — 9-18 digits (context-gated)
        self._bank_acct_re = re.compile(
            r'\b\d{9,18}\b',
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, text: str) -> str:
        """Return *text* with all detected PII replaced by redaction tags."""
        if not text:
            return text

        # UPI before email (avoids mis-classifying UPI as email)
        text = self._upi_re.sub('[UPI_REDACTED]', text)
        text = self._email_re.sub('[EMAIL_REDACTED]', text)
        text = self._aadhaar_re.sub('[AADHAAR_REDACTED]', text)
        text = self._pan_re.sub('[PAN_REDACTED]', text)
        text = self._ifsc_re.sub('[IFSC_REDACTED]', text)

        # Ethiopian phone (before generic patterns)
        text = self._phone_et_re.sub('[PHONE_REDACTED]', text)

        # Indian phone with +91
        text = self._mask_intl_indian_phone(text)

        # Bare 10-digit Indian phone with agri-context guard
        text = self._mask_bare_phone(text)

        # Ethiopian National ID
        text = self._eth_natid_re.sub('[NATIONAL_ID_REDACTED]', text)

        # Bank account (context-gated)
        text = self._mask_bank_account(text)

        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_intl_indian_phone(self, text: str) -> str:
        """Mask +91 prefixed phone numbers."""
        def _replace(m):
            matched = m.group()
            digits_only = re.sub(r'[\s\-]', '', matched)
            if digits_only.startswith('+91') or (digits_only.startswith('91') and len(digits_only) == 12):
                return '[PHONE_REDACTED]'
            return matched

        return self._phone_intl_in_re.sub(_replace, text)

    def _mask_bare_phone(self, text: str) -> str:
        """Mask 10-digit phone numbers starting with 6-9, skipping agri context."""
        def _replace(m):
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            window = text[start:end]
            if _AGRI_CONTEXT_WORDS.search(window):
                return m.group()
            return '[PHONE_REDACTED]'

        return self._phone_bare_re.sub(_replace, text)

    def _mask_bank_account(self, text: str) -> str:
        """Mask 9-18 digit numbers only when near banking keywords."""
        def _replace(m):
            # Skip if already redacted (inside a tag)
            if text[max(0, m.start() - 1):m.start()].endswith('['):
                return m.group()
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            window = text[start:end]
            if _BANK_CONTEXT_WORDS.search(window):
                return '[BANK_ACCOUNT_REDACTED]'
            return m.group()

        return self._bank_acct_re.sub(_replace, text)


# Global singleton
pii_masker = PIIMasker()
