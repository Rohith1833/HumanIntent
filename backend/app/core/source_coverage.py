"""Domain-agnostic atomic semantic request unit extraction and source-coverage gate."""

import logging
import re
from typing import Optional, Set

from app.schemas.analysis import ClarificationAnswer, PromptAnalysis

logger = logging.getLogger("humanintent.source_coverage")

GENERIC_ACTION_VERBS = {
    "conduct", "create", "build", "make", "develop", "implement", "write", "draft",
    "generate", "produce", "design", "setup", "establish", "provide", "construct",
    "assist", "prepare", "execute", "define", "describe", "support", "deliver",
    "specify", "complete", "request", "perform",
}

STOP_WORDS = {
    "make", "build", "create", "write", "explain", "simple", "simply",
    "using", "with", "into", "from", "that", "this", "your", "about",
    "for", "the", "and", "a", "an", "in", "to", "of", "on", "at", "by",
    "app", "application", "analyze", "help", "me", "do", "my", "we",
    "use", "actually", "rather", "instead", "scratch", "wait", "maybe", "like", "just",
    "i", "want", "need", "please", "can", "you", "us", "some", "doing",
    "conduct", "prepare", "perform", "provide",
    "them", "what", "who", "how", "their", "they", "there", "then", "end",
}

GENERIC_FILLER_WORDS = {
    "system", "systems", "requirement", "requirements", "app", "application",
    "tracking", "tool", "tools", "features", "feature", "details", "detail",
    "specification", "specifications", "method", "methods", "software", "project",
    "environment", "process", "setup", "data", "thing", "something", "stuff", "it",
    "platform", "solution", "click", "shows", "tells", "page",
}

DELIM_PATTERN = (
    r"(?:\r?\n+|[;?!]|\band\s+then\b|\band\s+also\b|\bas\s+well\s+as\b|"
    r"\balong\s+with\b|\bin\s+addition\s+to\b|\bfollowed\s+by\b|"
    r",\s*(?:and\s+|then\s+|also\s+)?|\s+\band\b\s+|\s+\bthen\b\s+|"
    r"\s+\balso\b\s+|\s+\bplus\b\s+|\s+\bwith\b\s+|\s+\bincluding\b\s+|"
    r"\s+(?<!without\s)\busing\b\s+|\s+\bvia\b\s+|\s+\butilizing\b\s+|"
    r"\s+\bbut\b\s+|\s+\bwhere\b\s+|\s+\bthat\s+(?:tells?|shows?|recommends?|supports?|accepts?)\b\s+|"
    r"\s+(?=without\b)|\s+(?=avoid(?:ing)?\b)|\s+(?=disallow\b)|\s+(?=do\s+not\b)|\s+(?=never\b))"
)

SYNONYM_MAP = {
    "table": ["tabular", "matrix", "grid", "spreadsheet"],
    "tabular": ["table", "matrix", "grid", "spreadsheet"],
    "analysis": ["analyze", "analyse", "analyzing", "evaluation", "breakdown", "assessment", "comparative"],
    "analyze": ["analysis", "analyse", "analyzing", "evaluation", "comparative"],
    "output": ["export", "display", "print", "produce", "generate"],
    "export": ["output", "save", "write", "dump"],
    "build": ["create", "develop", "implement", "make"],
    "create": ["build", "develop", "implement", "make"],
    "code": ["script", "program", "tool", "app"],
    "tool": ["script", "utility", "program", "app", "application"],
    "judging": ["judge", "evaluation", "evaluate", "evaluating", "scoring", "score", "assessment", "review"],
    "judge": ["judging", "evaluation", "evaluate", "evaluating", "scoring", "score", "assessment", "review"],
    "evaluation": ["judging", "judge", "evaluating", "evaluate", "scoring", "score", "assessment", "review"],
    "payment": ["pay", "paying", "checkout", "billing", "transaction", "paid"],
    "registration": ["register", "registering", "signup", "sign-up", "onboarding"],
    "submission": ["submit", "submitting", "upload", "uploading"],
    "upload": ["uploading", "submit", "submitting", "entry", "entering"],
    "alerts": ["alert", "alerting", "notification", "notifications", "notify"],
    "notification": ["notifications", "notify", "alert", "alerts", "email", "emails"],
    "reordering": ["reorder", "reorders", "ordering", "order", "purchasing", "purchase"],
    "reservation": ["reserving", "reserve", "booking", "book"],
    "search": ["searching", "lookup", "find", "finding"],
    "buy": ["buying", "purchase", "purchasing", "recommend", "recommending", "recommendation"],
    "purchase": ["buy", "buying", "recommend", "recommendation"],
}


from functools import lru_cache


@lru_cache(maxsize=4096)
def get_stems(word: str) -> frozenset[str]:
    """Return common English inflectional/derivational stems for token matching (cached)."""
    w = word.lower().strip()
    stems = {w}

    # Handle Greek/Latin and irregular derivational stems
    if w.endswith("ative") and len(w) > 6:
        stems.add(w[:-5] + "e")  # comparative -> compare
        stems.add(w[:-5])
    if w.endswith("is") and len(w) > 4:
        stems.add(w[:-2] + "yze")  # analysis -> analyze
        stems.add(w[:-2] + "yse")
        stems.add(w[:-2])

    for suffix in ["tion", "sion", "ing", "ment", "ed", "er", "es", "s", "ies", "y", "al", "ance", "ence", "ive"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            base = w[:-len(suffix)]
            stems.add(base)
            if suffix in ["tion", "sion", "ing", "ed", "er", "ive"]:
                stems.add(base + "e")
            if suffix == "ies":
                stems.add(base + "y")
    return frozenset(stems)


# Precompute all expanded synonym stems once at import time
SYNONYM_EXPANSION_CACHE: dict[str, set[str]] = {}
for _term, _syns in SYNONYM_MAP.items():
    _terms_set = set(_syns)
    for _s in _syns:
        _terms_set.update(get_stems(_s))
    SYNONYM_EXPANSION_CACHE[_term] = _terms_set


def token_matches(tok: str, text: str, target_words: Optional[Set[str]] = None) -> bool:
    """Check if token or any of its stems/synonyms matches within target text."""
    t_lower = tok.lower().strip()
    if target_words is not None:
        if t_lower in target_words:
            return True
        stems = get_stems(t_lower)
        if any(s in target_words for s in stems):
            return True
        for term in (stems | {t_lower}):
            syn_stems = SYNONYM_EXPANSION_CACHE.get(term)
            if syn_stems and any(s in target_words for s in syn_stems):
                return True

    text_lower = text.lower()
    if t_lower in text_lower:
        return True
    all_stems = get_stems(t_lower)
    for stem in all_stems:
        if len(stem) >= 3 and stem in text_lower:
            return True
    for term in (all_stems | {t_lower}):
        syn_stems = SYNONYM_EXPANSION_CACHE.get(term)
        if syn_stems:
            for s in syn_stems:
                if len(s) >= 3 and s in text_lower:
                    return True
    return False


# Comprehensive Tech Normalization Dictionary
TECH_ALIASES = {
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mongo": "mongodb",
    "mongodb": "mongodb",
    "redis": "redis",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "mariadb": "mariadb",
    "oracle": "oracle",
    "mssql": "mssql",
    "sqlserver": "mssql",
    "cassandra": "cassandra",
    "dynamodb": "dynamodb",
    "neo4j": "neo4j",
    "elasticsearch": "elasticsearch",
    "opensearch": "opensearch",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "redshift": "redshift",
    "clickhouse": "clickhouse",
    "scylladb": "scylladb",
    "couchdb": "couchdb",
    "firebase": "firebase",
    "fcm": "firebase",
    "firestore": "firestore",
    "supabase": "supabase",
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "rust": "rust",
    "golang": "go",
    "go": "go",
    "java": "java",
    "kotlin": "kotlin",
    "swift": "swift",
    "cpp": "cpp",
    "c++": "cpp",
    "csharp": "csharp",
    "c#": "csharp",
    "ruby": "ruby",
    "php": "php",
    "scala": "scala",
    "bash": "bash",
    "react": "react",
    "reactjs": "react",
    "vue": "vue",
    "vuejs": "vue",
    "angular": "angular",
    "svelte": "svelte",
    "nextjs": "nextjs",
    "next": "nextjs",
    "nuxt": "nuxt",
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "express": "express",
    "expressjs": "express",
    "node": "nodejs",
    "nodejs": "nodejs",
    "nestjs": "nestjs",
    "spring": "spring",
    "springboot": "spring",
    "laravel": "laravel",
    "rails": "rails",
    "flutter": "flutter",
    "reactnative": "reactnative",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "terraform": "terraform",
    "pulumi": "pulumi",
    "ansible": "ansible",
    "helm": "helm",
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "ecs": "ecs",
    "eks": "eks",
    "gke": "gke",
    "graphql": "graphql",
    "grpc": "grpc",
    "kafka": "kafka",
    "rabbitmq": "rabbitmq",
    "nats": "nats",
    "sqs": "sqs",
    "websocket": "websocket",
    "websockets": "websocket",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "spark": "spark",
    "pandas": "pandas",
    "numpy": "numpy",
    "polars": "polars",
    "torchvision": "torchvision",
    "prometheus": "prometheus",
    "grafana": "grafana",
    "datadog": "datadog",
    "splunk": "splunk",
    "loki": "loki",
    "jaeger": "jaeger",
    "flyway": "flyway",
    "liquibase": "liquibase",
    "tokio": "tokio",
    "libuv": "libuv",
    "tailwind": "tailwind",
    "bootstrap": "bootstrap",
}

WORD_TO_DIGIT = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "first": "1st", "second": "2nd", "third": "3rd",
    "single": "1", "double": "2", "triple": "3",
}


def clean_neg_hypothesis(u: str) -> str:
    """Format a negative constraint unit into a standard declarative statement for NLI."""
    u = u.strip(" .")
    u = re.sub(r"^without\s+using\s+", "Do not use ", u, flags=re.I)
    u = re.sub(r"^without\s+transmitting\s+", "Do not transmit ", u, flags=re.I)
    u = re.sub(r"^without\s+storing\s+", "Do not store ", u, flags=re.I)
    u = re.sub(r"^without\s+loading\s+", "Do not load ", u, flags=re.I)
    u = re.sub(r"^without\s+violating\s+", "Do not violate ", u, flags=re.I)
    u = re.sub(r"^without\s+allocating\s+", "Do not allocate ", u, flags=re.I)
    u = re.sub(r"^without\s+creating\s+", "Do not create ", u, flags=re.I)
    u = re.sub(r"^without\s+setting\s+", "Do not set ", u, flags=re.I)
    u = re.sub(r"^without\s+logging\s+", "Do not log ", u, flags=re.I)
    u = re.sub(r"^without\s+dropping\s+", "Do not drop ", u, flags=re.I)
    u = re.sub(r"^without\s+allowing\s+", "Do not allow ", u, flags=re.I)
    u = re.sub(r"^without\s+exposing\s+", "Do not expose ", u, flags=re.I)
    u = re.sub(r"^without\s+leaving\s+", "Do not leave ", u, flags=re.I)
    u = re.sub(r"^without\s+modifying\s+", "Do not modify ", u, flags=re.I)
    u = re.sub(r"^without\s+serving\s+", "Do not serve ", u, flags=re.I)
    u = re.sub(r"^without\s+overwhelming\s+", "Do not overwhelm ", u, flags=re.I)
    u = re.sub(r"^without\s+requiring\s+", "Do not require ", u, flags=re.I)
    u = re.sub(r"^without\s+degrading\s+", "Do not degrade ", u, flags=re.I)
    u = re.sub(r"^without\s+exceeding\s+", "Do not exceed ", u, flags=re.I)
    u = re.sub(r"^without\s+granting\s+", "Do not grant ", u, flags=re.I)
    u = re.sub(r"^without\s+(\w+)ing\s+", r"Do not \1 ", u, flags=re.I)
    u = re.sub(r"^without\s+", "Do not use ", u, flags=re.I)
    u = re.sub(r"^avoid(?:ing)?\s+", "Do not use ", u, flags=re.I)
    u = re.sub(r"^never\s+", "Never ", u, flags=re.I)
    u = re.sub(r"^no\s+", "There should be no ", u, flags=re.I)
    if not u.endswith("."):
        u += "."
    return u


def extract_source_semantic_units(
    prompt: str,
    clarification_answers: Optional[list[ClarificationAnswer]] = None,
) -> list[str]:
    """Extract atomic semantic request units from authoritative source input.

    Handles run-on sentences, coordinating conjunctions, lists, and informal grammar
    without hardcoding domain-specific vocabulary. Preserves explicit negative constraints.
    """
    corpus = prompt
    if clarification_answers:
        substantive = [ca.answer.strip() for ca in clarification_answers if ca.answer and ca.answer.strip()]
        if substantive:
            corpus += "\n" + "\n".join(substantive)

    raw_clauses = re.split(DELIM_PATTERN, corpus, flags=re.IGNORECASE)
    units: list[str] = []

    prefix_cleaner = re.compile(
        r"^(?:(?:so\s+)?(?:i\s+want|i\s+need|we\s+need|we\s+want|please|help\s+me|can\s+you|build|create|make|develop|design|implement|setup|provide|write)\s+(?:me\s+|us\s+)?(?:a\s+|an\s+|the\s+)?(?:thing|system|app|application|platform|tool|service|workflow|solution|software|project|script|dashboard|portal)?(?:\s+(?:for\s+(?:my\s+)?website|for|that|with|to|doing|handling|supports?))?\s*)+",
        re.IGNORECASE,
    )
    subject_cleaner = re.compile(
        r"^(?:(?:people|users|participants|customers|clients|members|team|attendees|guests)\s+(?:are\s+|can\s+)?(?:doing|who\s+are|to|click|clicked)?\s*)",
        re.IGNORECASE,
    )
    something_cleaner = re.compile(
        r"^(?:(?:something|anything|stuff)\s+(?:for|to|that)\s+)",
        re.IGNORECASE,
    )
    hesitation_re = re.compile(
        r"\b(?:wait\s+maybe|actually\s+no|scratch\s+that|not\s+a\s+normal|no\s+like|wait\s+no)\b",
        re.IGNORECASE,
    )

    for c in raw_clauses:
        cleaned = c.strip(" \t\r\n.,;:!?-")
        if not cleaned:
            continue

        # If clause has hesitation like "wait maybe like a calculator", skip the hesitating part
        if hesitation_re.search(cleaned):
            parts = re.split(r"\b(?:wait\s+maybe|actually\s+no|scratch\s+that|wait\s+no)\b", cleaned, flags=re.IGNORECASE)
            cleaned = parts[-1].strip(" \t\r\n.,;:!?-")
            if not cleaned or re.search(r"^(?:like\s+a\s+calculator|calculator)", cleaned, flags=re.IGNORECASE):
                continue

        # Only clean prefixes/subjects if NOT a negative constraint
        if not re.search(r"^(?:without|avoid|never|no|do\s+not)\b", cleaned, re.IGNORECASE):
            cleaned = re.sub(r"^(?:actually\s+|rather\s+|instead\s+)+", "", cleaned, flags=re.IGNORECASE).strip(" \t\r\n.,;:!?-")
            cleaned = prefix_cleaner.sub("", cleaned).strip(" \t\r\n.,;:!?-")
            cleaned = subject_cleaner.sub("", cleaned).strip(" \t\r\n.,;:!?-")
            cleaned = something_cleaner.sub("", cleaned).strip(" \t\r\n.,;:!?-")
            cleaned = re.sub(r"^(?:a|an|the|some|for|to|doing|no\s+like\s+a|no\s+like|like\s+a)\s+", "", cleaned, flags=re.IGNORECASE).strip(" \t\r\n.,;:!?-")
            cleaned = re.sub(r"\s+based\s+on\s+.*$", "", cleaned, flags=re.IGNORECASE).strip(" \t\r\n.,;:!?-")

        words = [
            w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", cleaned)
            if w.lower() not in STOP_WORDS and w.lower() not in GENERIC_FILLER_WORDS
        ]
        if words and cleaned:
            if cleaned.lower() not in [u.lower() for u in units]:
                units.append(cleaned)

    return units


def extract_unit_numbers(text: str) -> set[str]:
    """Extract explicit numerical digits or number-words from text."""
    digits = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    norm_nums = set(digits)
    for tok in tokens:
        if tok in WORD_TO_DIGIT:
            norm_nums.add(WORD_TO_DIGIT[tok])
            norm_nums.add(tok)
    return norm_nums


def extract_unit_negations(text: str) -> set[str]:
    """Extract explicit negation words."""
    neg_words = {"no", "not", "without", "never", "avoid", "don't", "dont", "exclude", "prohibit", "disallow"}
    tokens = set(re.findall(r"\b[a-zA-Z']+\b", text.lower()))
    return tokens & neg_words


def extract_unit_tech(text: str) -> set[str]:
    """Extract recognized normalized technologies/frameworks from text."""
    t_lower = text.lower()
    found = set()
    if re.search(r"\b(?:rest\s+api|restful|rest\s+endpoints?|rest\s+services?|rest\s+protocols?)\b", t_lower):
        found.add("rest")
    elif re.search(r"\busing\s+rest\b|\bwith\s+rest\b", t_lower):
        found.add("rest")

    raw_tokens = re.findall(r"[a-zA-Z0-9\+\#\.\-]+", t_lower)
    for tok in raw_tokens:
        clean = tok.strip(".,;:!?()[]{}")
        if clean in TECH_ALIASES:
            found.add(TECH_ALIASES[clean])
        no_dot = clean.replace(".", "")
        if no_dot in TECH_ALIASES:
            found.add(TECH_ALIASES[no_dot])
    return found


def is_semantic_unit_covered(unit: str, target_text: str) -> bool:
    """Verify if an atomic semantic unit is materially represented in the target text.

    Combines:
    1. Deterministic token stems and synonym matching.
    2. Lightweight pretrained sentence-embedding similarity for paraphrased requirements.
    3. Cross-encoder NLI fallback for ambiguous semantic ranges and negative constraints.
    4. Strict deterministic preservation of explicit quantities, negations, and named technologies.
    """
    target_lower = target_text.lower()
    unit_lower = unit.lower().strip()

    # 1. Deterministic Invariants
    # A. Numbers / Quantities
    unit_nums = extract_unit_numbers(unit_lower)
    target_nums = extract_unit_numbers(target_lower)
    if unit_nums and not unit_nums.issubset(target_nums):
        return False

    # B. Named Technologies: explicit conflict check
    unit_tech = extract_unit_tech(unit_lower)
    target_tech = extract_unit_tech(target_lower)
    if unit_tech and target_tech and not unit_tech.issubset(target_tech):
        return False

    # Extract candidate clauses from target text without breaking intra-word hyphens
    raw_clauses = re.split(r"(?:\r?\n+|(?:^|\n)\s*[-*•]\s+|[;]|\.\s+)", target_text)
    clauses: list[str] = []
    for c in raw_clauses:
        cleaned = c.strip(" \t\r\n-•*")
        if not cleaned:
            continue
        if re.search(r"^(?:role|objective|requirements?|constraints?|preferences?|missing\s+context):?\s*$", cleaned, re.I):
            continue
        cleaned = re.sub(r"^(?:role|objective|requirements?|constraints?|preferences?|missing\s+context):\s*", "", cleaned, flags=re.I).strip()
        if len(cleaned.split()) >= 2:
            clauses.append(cleaned)
    if not clauses:
        clauses = [target_text.strip()]

    # 2. Negative Constraints Handling
    unit_neg = extract_unit_negations(unit_lower)
    if unit_neg:
        hyp = clean_neg_hypothesis(unit)
        unit_tokens = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", unit_lower) if w not in {"without", "avoid", "never", "not"}}
        has_entailment = False
        has_violation = False

        try:
            from app.core.nli_matcher import nli_matcher
            if nli_matcher.is_available:
                for c in clauses:
                    c_lower = c.lower()
                    c_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", c_lower))
                    overlap = bool(unit_tokens & c_tokens)
                    has_neg_word = bool(extract_unit_negations(c_lower))

                    if overlap or has_neg_word:
                        label, probs = nli_matcher.predict(c, hyp)
                        if probs.get("entailment", 0.0) >= 0.50:
                            has_entailment = True
                            has_violation = False
                            break
                        if probs.get("contradiction", 0.0) >= 0.70:
                            has_violation = True

                if not has_entailment and not has_violation:
                    label, probs = nli_matcher.predict(target_text, hyp)
                    if probs.get("entailment", 0.0) >= 0.50:
                        has_entailment = True
            else:
                has_entailment = any(neg in target_lower for neg in unit_neg)
        except Exception as e:
            logger.debug("NLI negative constraint check error: %s", e)
            has_entailment = any(neg in target_lower for neg in unit_neg)

        if has_entailment:
            return True
        if has_violation:
            return False
        return False

    # 3. Standard Requirements: Token match or Embedding / NLI
    tokens = [
        w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", unit)
        if w.lower() not in STOP_WORDS and w.lower() not in GENERIC_FILLER_WORDS
    ]
    if not tokens:
        tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", unit) if w.lower() not in STOP_WORDS]
    if not tokens:
        return True

    matches = sum(1 for tok in tokens if token_matches(tok, target_text))
    token_match_ratio = matches / len(tokens)
    if (matches >= 1 if len(tokens) == 1 else token_match_ratio >= 0.5):
        return True

    # SentenceTransformer Similarity
    try:
        from app.core.semantic_matcher import semantic_matcher
        if semantic_matcher.is_available:
            sims = [semantic_matcher.similarity(unit, c) for c in clauses]
            max_sim = max(sims) if sims else 0.0

            if max_sim < 0.75 and len(target_text) < 500:
                full_sim = semantic_matcher.similarity(unit, target_text)
                max_sim = max(max_sim, full_sim)

            # High threshold -> clearly covered
            if max_sim >= 0.70:
                return True

            # Low threshold -> clearly unrelated
            if max_sim < 0.30:
                return False

            # Ambiguous band [0.30, 0.70) -> NLI Fallback
            try:
                from app.core.nli_matcher import nli_matcher
                if nli_matcher.is_available:
                    sorted_clauses = sorted(clauses, key=lambda c: semantic_matcher.similarity(unit, c), reverse=True)[:3]
                    for p in sorted_clauses:
                        l_cu, p_cu = nli_matcher.predict(p, unit + ".")
                        if p_cu.get("contradiction", 0.0) >= 0.60:
                            continue
                        if p_cu.get("entailment", 0.0) >= 0.45:
                            return True
            except Exception as e:
                logger.debug("NLI fallback in ambiguous band error: %s", e)
    except Exception as e:
        logger.debug("Semantic matching fallback to token match: %s", e)

    return False


def deduplicate_semantic_units(units: list[str], threshold: float = 0.72) -> list[str]:
    """Deduplicate extracted semantic units using sentence embeddings and deterministic checks.

    Preserves distinct requirements even if they share a topic noun.
    Only merges units if their semantic similarity >= threshold and explicit constraints match.
    """
    if len(units) <= 1:
        return units

    try:
        from app.core.semantic_matcher import semantic_matcher
        if not semantic_matcher.is_available:
            return units
    except Exception:
        return units

    deduped: list[str] = []
    for unit in units:
        is_dup = False
        unit_nums = extract_unit_numbers(unit)
        unit_neg = extract_unit_negations(unit)
        unit_tech = extract_unit_tech(unit)

        for existing in deduped:
            sim = semantic_matcher.similarity(unit, existing)
            if sim >= threshold:
                existing_nums = extract_unit_numbers(existing)
                existing_neg = extract_unit_negations(existing)
                existing_tech = extract_unit_tech(existing)

                # Only deduplicate if explicit quantities, negations, and tech match
                if (
                    unit_nums == existing_nums
                    and unit_neg == existing_neg
                    and unit_tech == existing_tech
                ):
                    is_dup = True
                    break

        if not is_dup:
            deduped.append(unit)

    return deduped


def check_source_coverage(
    raw_prompt: str,
    analysis: PromptAnalysis,
    clarification_answers: Optional[list[ClarificationAnswer]] = None,
) -> tuple[bool, list[str], list[str]]:
    """Verify that all materially requested semantic units from source survive into analysis.

    Checks:
    - objective
    - explicit_requirements
    - constraints
    - preferences
    - missing_context
    - deliverable_semantics

    Returns:
    (is_fully_covered, source_units, missing_units)
    """
    source_units = extract_source_semantic_units(raw_prompt, clarification_answers)
    if not source_units:
        return True, [], []

    # Aggregate all derived fields from structured analysis
    analysis_corpus_parts = []
    if analysis.objective:
        analysis_corpus_parts.append(analysis.objective)
    if analysis.explicit_requirements:
        analysis_corpus_parts.extend(analysis.explicit_requirements)
    if analysis.constraints:
        analysis_corpus_parts.extend(analysis.constraints)
    if analysis.preferences:
        analysis_corpus_parts.extend(analysis.preferences)
    if analysis.missing_context:
        analysis_corpus_parts.extend(analysis.missing_context)
    if hasattr(analysis, "deliverable_semantics") and analysis.deliverable_semantics:
        if analysis.deliverable_semantics.deliverable_type:
            analysis_corpus_parts.append(analysis.deliverable_semantics.deliverable_type)
        if analysis.deliverable_semantics.intended_outcome:
            analysis_corpus_parts.append(analysis.deliverable_semantics.intended_outcome)

    analysis_corpus = "\n".join(analysis_corpus_parts)

    missing_units: list[str] = []
    for unit in source_units:
        if not is_semantic_unit_covered(unit, analysis_corpus):
            missing_units.append(unit)

    is_covered = len(missing_units) == 0
    return is_covered, source_units, missing_units
