import re
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class SanitizeResult(BaseModel):
    """Result of privacy and secret sanitization on a prompt string."""
    text: str
    sanitized: bool = False
    pii_types_removed: List[str] = Field(default_factory=list)
    is_secret_risk: bool = False
    risk_reason: Optional[str] = None


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)
KNOWN_TOKEN_PREFIX_PATTERN = re.compile(
    r"\b(?:sk|pk|ghp|gho|ghu|ghs|ghr|xoxb|xoxp|sq0csp)_[a-zA-Z0-9_\-]{16,}\b"
)
GENERIC_API_KEY_PATTERN = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?"
)
BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b"
)
PASSWORD_PATTERN = re.compile(
    r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?"
)
AUTH_URL_PATTERN = re.compile(
    r"https?://[^\s]+[?&](?:token|key|secret|password|auth|sig|signature)=[^\s&]+"
)
PRIVATE_IP_PATTERN = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b"
)
HIGH_RISK_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
)


def sanitize_text(text: str) -> SanitizeResult:
    """Sanitize PII and credentials from a prompt string.

    Detects emails, phones, API keys, passwords, bearer tokens, authenticated URLs,
    and private IPs. Excludes records containing unredactable high-risk secrets.
    """
    if not text:
        return SanitizeResult(text="", sanitized=False)

    # 1. High-risk secret detection: unredactable private keys
    if HIGH_RISK_PRIVATE_KEY_BLOCK.search(text):
        return SanitizeResult(
            text=text,
            sanitized=False,
            is_secret_risk=True,
            risk_reason="Unredactable private cryptographic key block detected",
        )

    current_text = text
    pii_types_removed: List[str] = []

    # 2. Authenticated URLs
    if AUTH_URL_PATTERN.search(current_text):
        current_text = AUTH_URL_PATTERN.sub("<URL>", current_text)
        pii_types_removed.append("AUTH_URL")

    # 3. Known API Key prefixes (OpenAI, GitHub, Slack, Stripe)
    if KNOWN_TOKEN_PREFIX_PATTERN.search(current_text):
        current_text = KNOWN_TOKEN_PREFIX_PATTERN.sub("<API_KEY>", current_text)
        pii_types_removed.append("API_KEY")

    # 4. Bearer tokens
    if BEARER_TOKEN_PATTERN.search(current_text):
        current_text = BEARER_TOKEN_PATTERN.sub("Bearer <API_KEY>", current_text)
        if "API_KEY" not in pii_types_removed:
            pii_types_removed.append("API_KEY")

    # 5. Generic API Key key=value patterns
    if GENERIC_API_KEY_PATTERN.search(current_text):
        current_text = GENERIC_API_KEY_PATTERN.sub(lambda m: m.group(0).replace(m.group(1), "<API_KEY>"), current_text)
        if "API_KEY" not in pii_types_removed:
            pii_types_removed.append("API_KEY")

    # 6. Passwords
    if PASSWORD_PATTERN.search(current_text):
        current_text = PASSWORD_PATTERN.sub(lambda m: m.group(0).replace(m.group(1), "<PASSWORD>"), current_text)
        pii_types_removed.append("PASSWORD")

    # 7. Emails
    if EMAIL_PATTERN.search(current_text):
        current_text = EMAIL_PATTERN.sub("<EMAIL>", current_text)
        pii_types_removed.append("EMAIL")

    # 8. SSN / Credit Cards
    if SSN_PATTERN.search(current_text):
        current_text = SSN_PATTERN.sub("<ID>", current_text)
        pii_types_removed.append("SSN")

    if CREDIT_CARD_PATTERN.search(current_text):
        current_text = CREDIT_CARD_PATTERN.sub("<CREDIT_CARD>", current_text)
        pii_types_removed.append("CREDIT_CARD")

    # 9. Phones
    if PHONE_PATTERN.search(current_text):
        # Avoid masking normal software version numbers like 1.2.3 or 10.0.0
        current_text = PHONE_PATTERN.sub("<PHONE>", current_text)
        pii_types_removed.append("PHONE")

    # 10. Private IPs
    if PRIVATE_IP_PATTERN.search(current_text):
        current_text = PRIVATE_IP_PATTERN.sub("<IP_ADDRESS>", current_text)
        pii_types_removed.append("PRIVATE_IP")

    sanitized = bool(pii_types_removed)
    return SanitizeResult(
        text=current_text,
        sanitized=sanitized,
        pii_types_removed=sorted(list(set(pii_types_removed))),
        is_secret_risk=False,
    )
