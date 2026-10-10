"""
Benchmark generator and freeze utility for Phase 10.
Produces 220 diverse, curated benchmark cases across all 11 PromptCategory values,
covering behavioral variants, difficulty distributions, and invariant annotations.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkExpected,
    BenchmarkMetadata,
    BenchmarkProvenance,
    EvaluationDifficulty,
)


def generate_benchmark_cases() -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []

    # Category 1: CODING (20 cases)
    coding_specs = [
        ("cod_001", EvaluationDifficulty.EASY, "Write a Python function to validate whether a given string is a valid IPv4 address without using the ipaddress library.",
         {"task_family": "CODING", "must_preserve": ["IPv4 address", "Python", "without ipaddress library"], "must_not_invent": ["IPv6 support", "DNS resolution"], "constraints": ["without using the ipaddress library"], "quantities": []}),
        ("cod_002", EvaluationDifficulty.MEDIUM, "Implement an LRU Cache in TypeScript with get and put operations in O(1) time complexity. Avoid third-party packages and include types.",
         {"task_family": "CODING", "must_preserve": ["LRU Cache", "TypeScript", "O(1) time complexity", "get and put"], "must_not_invent": ["Redis backend", "disk persistence"], "constraints": ["avoid third-party packages"], "quantities": []}),
        ("cod_003", EvaluationDifficulty.HARD, "Create a concurrent worker pool in Go that processes jobs from a buffered channel with a max limit of 10 workers and handles graceful shutdown on SIGINT.",
         {"task_family": "CODING", "must_preserve": ["worker pool", "Go", "buffered channel", "graceful shutdown", "SIGINT"], "must_not_invent": ["HTTP REST server", "gRPC"], "constraints": ["max limit of 10 workers"], "quantities": ["10 workers"]}),
        ("cod_004", EvaluationDifficulty.EASY, "Write a SQL query in PostgreSQL to find all customers who made more than 3 purchases in the last 30 days.",
         {"task_family": "CODING", "must_preserve": ["SQL query", "PostgreSQL", "more than 3 purchases", "last 30 days"], "must_not_invent": ["database migration script"], "constraints": [], "quantities": ["3 purchases", "30 days"]}),
        ("cod_005", EvaluationDifficulty.MEDIUM, "Build a custom React hook `useDebounce` in TypeScript that delays updating a value by 300ms. No external libraries.",
         {"task_family": "CODING", "must_preserve": ["React hook", "useDebounce", "TypeScript", "300ms"], "must_not_invent": ["Lodash"], "constraints": ["no external libraries"], "quantities": ["300ms"]}),
        ("cod_006", EvaluationDifficulty.HARD, "Implement a lock-free single-producer single-consumer ring buffer in C++20 using std::atomic. Do not use mutexes.",
         {"task_family": "CODING", "must_preserve": ["ring buffer", "C++20", "std::atomic", "single-producer single-consumer"], "must_not_invent": ["POSIX threads lock"], "constraints": ["do not use mutexes"], "quantities": []}),
        ("cod_007", EvaluationDifficulty.EASY, "Create a bash script to recursively search a directory for all .log files older than 7 days and delete them. Run dry-run by default.",
         {"task_family": "CODING", "must_preserve": ["bash script", ".log files", "older than 7 days", "delete", "dry-run by default"], "must_not_invent": ["AWS S3 upload"], "constraints": ["dry-run by default"], "quantities": ["7 days"]}),
        ("cod_008", EvaluationDifficulty.MEDIUM, "Write a Rust function that computes the Levenshtein distance between two string slices without allocating a full matrix in memory.",
         {"task_family": "CODING", "must_preserve": ["Rust", "Levenshtein distance", "two string slices"], "must_not_invent": ["Damerau-Levenshtein", "regex engine"], "constraints": ["without allocating a full matrix"], "quantities": []}),
        ("cod_009", EvaluationDifficulty.MEDIUM, "Implement a Trie data structure in Java supporting insert, search, and startsWith operations. Only use standard java.util.",
         {"task_family": "CODING", "must_preserve": ["Trie", "Java", "insert", "search", "startsWith"], "must_not_invent": ["Apache Commons"], "constraints": ["standard java.util only"], "quantities": []}),
        ("cod_010", EvaluationDifficulty.HARD, "Design a distributed snowflake ID generator in Kotlin using 41 bits timestamp, 10 bits machine ID, and 12 bits sequence number. Must be thread-safe.",
         {"task_family": "CODING", "must_preserve": ["snowflake ID generator", "Kotlin", "41 bits timestamp", "10 bits machine ID", "12 bits sequence number", "thread-safe"], "must_not_invent": ["Zookeeper integration"], "constraints": ["must be thread-safe"], "quantities": ["41 bits", "10 bits", "12 bits"]}),
        ("cod_011", EvaluationDifficulty.EASY, "make a python script to fetch json from an api and print keys",
         {"task_family": "CODING", "must_preserve": ["Python script", "fetch JSON", "API", "print keys"], "must_not_invent": ["FastAPI server"], "constraints": [], "quantities": []}),
        ("cod_012", EvaluationDifficulty.MEDIUM, "Create a Dockerfile for a multi-stage Node.js 20 production build that minimizes image size and runs as a non-root user.",
         {"task_family": "CODING", "must_preserve": ["Dockerfile", "multi-stage", "Node.js 20", "minimizes image size", "non-root user"], "must_not_invent": ["Kubernetes helm chart"], "constraints": ["runs as a non-root user"], "quantities": []}),
        ("cod_013", EvaluationDifficulty.MEDIUM, "Write an async Python function using aiohttp to download 5 URLs concurrently with a timeout of 10 seconds per request.",
         {"task_family": "CODING", "must_preserve": ["aiohttp", "download 5 URLs", "concurrently", "timeout 10 seconds"], "must_not_invent": ["BeautifulSoup web scraper"], "constraints": ["timeout of 10 seconds"], "quantities": ["5 URLs", "10 seconds"]}),
        ("cod_014", EvaluationDifficulty.HARD, "Implement an A* pathfinding algorithm on a 2D grid in C# with diagonal movement cost 1.414 and Manhattan heuristic for non-diagonal.",
         {"task_family": "CODING", "must_preserve": ["A* pathfinding", "2D grid", "C#", "diagonal movement", "heuristic"], "must_not_invent": ["Unity game engine specific API"], "constraints": [], "quantities": []}),
        ("cod_015", EvaluationDifficulty.EASY, "code an app",
         {"task_family": "CODING", "must_preserve": ["application"], "must_not_invent": ["React Native iOS app"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["target platform", "technology stack", "primary functionality"]}),
        ("cod_016", EvaluationDifficulty.MEDIUM, "Write a Python dataclass for an e-commerce Product with id, name, price in cents, and stock, with a method to check availability.",
         {"task_family": "CODING", "must_preserve": ["Python dataclass", "Product", "id", "name", "price in cents", "stock", "check availability"], "must_not_invent": ["Stripe checkout"], "constraints": [], "quantities": []}),
        ("cod_017", EvaluationDifficulty.MEDIUM, "Implement an Exponential Backoff retry utility function in TypeScript with max 4 retries and jitter.",
         {"task_family": "CODING", "must_preserve": ["Exponential Backoff", "TypeScript", "retry utility", "jitter", "max 4 retries"], "must_not_invent": ["Axios interceptor"], "constraints": ["max 4 retries"], "quantities": ["4 retries"]}),
        ("cod_018", EvaluationDifficulty.HARD, "Build a B-Tree search and insertion algorithm in pure Python without external dependencies, supporting order M=4.",
         {"task_family": "CODING", "must_preserve": ["B-Tree", "search and insertion", "Python", "order 4"], "must_not_invent": ["B+ Tree disk paging"], "constraints": ["without external dependencies"], "quantities": ["order 4"]}),
        ("cod_019", EvaluationDifficulty.EASY, "Write a CSS Grid layout with 3 equal columns on desktop and 1 column on screens under 768px.",
         {"task_family": "CODING", "must_preserve": ["CSS Grid layout", "3 equal columns", "desktop", "1 column", "under 768px"], "must_not_invent": ["Bootstrap 5"], "constraints": [], "quantities": ["3 columns", "1 column", "768px"]}),
        ("cod_020", EvaluationDifficulty.MEDIUM, "I want a regex to match valid semver version strings like 1.2.3 or 2.0.0-beta.1 in JavaScript.",
         {"task_family": "CODING", "must_preserve": ["regex", "semver version strings", "JavaScript"], "must_not_invent": ["npm package semver"], "constraints": [], "quantities": []}),
    ]

    # Category 2: DEBUGGING (20 cases)
    debugging_specs = [
        ("dbg_001", EvaluationDifficulty.MEDIUM, "My Python script crashes with `RecursionError: maximum recursion depth exceeded` when parsing a deep JSON tree. How do I fix it without increasing sys.setrecursionlimit?",
         {"task_family": "DEBUGGING", "must_preserve": ["RecursionError", "deep JSON tree", "Python"], "must_not_invent": ["switch to Go"], "constraints": ["without increasing sys.setrecursionlimit"], "quantities": []}),
        ("dbg_002", EvaluationDifficulty.HARD, "Investigate why a PostgreSQL query with `WHERE created_at >= NOW() - INTERVAL '7 days'` causes a sequential scan even though an index on created_at exists.",
         {"task_family": "DEBUGGING", "must_preserve": ["PostgreSQL", "sequential scan", "index on created_at", "WHERE created_at"], "must_not_invent": ["MongoDB migration"], "constraints": [], "quantities": ["7 days"]}),
        ("dbg_003", EvaluationDifficulty.EASY, "fix this code:\ndef add(a, b)\n  return a + b",
         {"task_family": "DEBUGGING", "must_preserve": ["def add(a, b)", "syntax error"], "must_not_invent": ["TypeScript compilation"], "constraints": [], "quantities": []}),
        ("dbg_004", EvaluationDifficulty.MEDIUM, "Why does `useEffect` in my React component run in an infinite loop when updating state inside it? Explain the root cause and provide the fix.",
         {"task_family": "DEBUGGING", "must_preserve": ["useEffect", "React", "infinite loop", "updating state", "root cause", "fix"], "must_not_invent": ["Redux Toolkit rewrite"], "constraints": [], "quantities": []}),
        ("dbg_005", EvaluationDifficulty.HARD, "Debug a data race in Go detected by `go test -race` where multiple goroutines access a shared map without synchronization.",
         {"task_family": "DEBUGGING", "must_preserve": ["data race", "Go", "go test -race", "shared map", "goroutines"], "must_not_invent": ["Kafka topic"], "constraints": [], "quantities": []}),
        ("dbg_006", EvaluationDifficulty.EASY, "My git push fails with 'rejected - non-fast-forward'. How do I resolve this safely without using git push --force?",
         {"task_family": "DEBUGGING", "must_preserve": ["git push", "non-fast-forward", "resolve safely"], "must_not_invent": ["delete git repository"], "constraints": ["without using git push --force"], "quantities": []}),
        ("dbg_007", EvaluationDifficulty.MEDIUM, "In Docker, my container exits immediately with code 137. What causes exit code 137 and how can I diagnose if it was an OOM kill?",
         {"task_family": "DEBUGGING", "must_preserve": ["Docker", "exit code 137", "OOM kill", "diagnose"], "must_not_invent": ["Kubernetes cluster deployment"], "constraints": [], "quantities": ["exit code 137"]}),
        ("dbg_008", EvaluationDifficulty.HARD, "A C++ program has a segmentation fault in std::vector::push_back during multi-threaded insertion. Explain iterator invalidation and thread safety issues.",
         {"task_family": "DEBUGGING", "must_preserve": ["C++", "segmentation fault", "std::vector::push_back", "multi-threaded", "iterator invalidation"], "must_not_invent": ["switch to Java"], "constraints": [], "quantities": []}),
        ("dbg_009", EvaluationDifficulty.MEDIUM, "My Node.js HTTP server hangs indefinitely when an upstream request times out. Show how to configure abort signals and timeouts in fetch.",
         {"task_family": "DEBUGGING", "must_preserve": ["Node.js", "upstream request timeout", "abort signals", "fetch"], "must_not_invent": ["Nginx load balancer config"], "constraints": [], "quantities": []}),
        ("dbg_010", EvaluationDifficulty.EASY, "Why is `0.1 + 0.2 === 0.3` false in JavaScript? Explain IEEE 754 floating point and how to compare numbers with an epsilon.",
         {"task_family": "DEBUGGING", "must_preserve": ["0.1 + 0.2", "JavaScript", "IEEE 754", "epsilon"], "must_not_invent": ["BigInt math"], "constraints": [], "quantities": []}),
        ("dbg_011", EvaluationDifficulty.MEDIUM, "Diagnose why a CSS sticky header is not sticking to the top of the viewport when scrolling down.",
         {"task_family": "DEBUGGING", "must_preserve": ["CSS sticky header", "position sticky", "not sticking", "viewport"], "must_not_invent": ["JavaScript scroll listener script"], "constraints": [], "quantities": []}),
        ("dbg_012", EvaluationDifficulty.HARD, "A Python FastAPI app has high memory usage that never drops after handling large file uploads. How to profile memory leaks using tracemalloc or memray?",
         {"task_family": "DEBUGGING", "must_preserve": ["Python FastAPI", "memory leak", "large file uploads", "profiling", "tracemalloc"], "must_not_invent": ["reboot server cron job"], "constraints": [], "quantities": []}),
        ("dbg_013", EvaluationDifficulty.EASY, "fix the bug",
         {"task_family": "DEBUGGING", "must_preserve": ["bug fix"], "must_not_invent": ["C++ buffer overflow patch"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["error message or symptoms", "source code snippet", "programming language"]}),
        ("dbg_014", EvaluationDifficulty.MEDIUM, "Fix a Django ORM N+1 query issue on a page listing 100 books and their authors using select_related or prefetch_related.",
         {"task_family": "DEBUGGING", "must_preserve": ["Django ORM", "N+1 query", "100 books", "select_related or prefetch_related"], "must_not_invent": ["GraphQL server"], "constraints": [], "quantities": ["100 books"]}),
        ("dbg_015", EvaluationDifficulty.HARD, "Troubleshoot a TLS handshake error 'certificate verify failed: self signed certificate in certificate chain' in Python requests on corporate network.",
         {"task_family": "DEBUGGING", "must_preserve": ["TLS handshake error", "self signed certificate", "Python requests", "corporate network"], "must_not_invent": ["disable SSL globally everywhere"], "constraints": [], "quantities": []}),
        ("dbg_016", EvaluationDifficulty.MEDIUM, "Why does `git rebase` cause merge conflicts on every single commit instead of once, and how do I abort safely?",
         {"task_family": "DEBUGGING", "must_preserve": ["git rebase", "merge conflicts", "every commit", "abort safely"], "must_not_invent": ["git cherry-pick loop"], "constraints": [], "quantities": []}),
        ("dbg_017", EvaluationDifficulty.EASY, "My bash script terminates with `syntax error: unexpected end of file`. What are the most common causes like unclosed quotes or heredocs?",
         {"task_family": "DEBUGGING", "must_preserve": ["bash script", "syntax error unexpected end of file", "unclosed quotes or heredocs"], "must_not_invent": ["rewrite in Perl"], "constraints": [], "quantities": []}),
        ("dbg_018", EvaluationDifficulty.MEDIUM, "Identify why `JSON.parse` throws `Unexpected token u in JSON at position 0` in browser JavaScript when reading from localStorage.",
         {"task_family": "DEBUGGING", "must_preserve": ["JSON.parse", "Unexpected token u", "JavaScript", "localStorage"], "must_not_invent": ["IndexedDB migration"], "constraints": [], "quantities": []}),
        ("dbg_019", EvaluationDifficulty.HARD, "Diagnose a Kubernetes CrashLoopBackOff error on a Pod where readinessProbe succeeds but livenessProbe fails after 30 seconds.",
         {"task_family": "DEBUGGING", "must_preserve": ["Kubernetes", "CrashLoopBackOff", "readinessProbe", "livenessProbe", "30 seconds"], "must_not_invent": ["Ansible playbook"], "constraints": [], "quantities": ["30 seconds"]}),
        ("dbg_020", EvaluationDifficulty.EASY, "Review this code and tell me what is wrong:\n```python\nfor i in range(len(items)):\n    if items[i] == target:\n        items.pop(i)\n```",
         {"task_family": "DEBUGGING", "must_preserve": ["items.pop(i)", "index out of range", "modifying list during iteration", "Python"], "must_not_invent": ["switch to pandas dataframe"], "constraints": [], "quantities": []}),
    ]

    # Category 3: WRITING (20 cases)
    writing_specs = [
        ("wri_001", EvaluationDifficulty.EASY, "Write a polite reminder email to a client regarding an unpaid invoice #4021 that is 14 days overdue. Keep it under 150 words.",
         {"task_family": "WRITING", "must_preserve": ["reminder email", "invoice #4021", "14 days overdue", "polite tone"], "must_not_invent": ["legal action threat"], "constraints": ["under 150 words"], "quantities": ["150 words", "14 days"]}),
        ("wri_002", EvaluationDifficulty.MEDIUM, "Draft a project status update email to the executive steering committee summarizing Q3 deliverables, key risks, and 2 mitigation steps. Corporate professional tone.",
         {"task_family": "WRITING", "must_preserve": ["status update email", "executive steering committee", "Q3 deliverables", "key risks", "2 mitigation steps"], "must_not_invent": ["quarterly earnings transcript"], "constraints": ["corporate professional tone"], "quantities": ["2 mitigation steps"]}),
        ("wri_003", EvaluationDifficulty.HARD, "Write a formal apology letter from a SaaS company CEO addressing a major 4-hour service outage, explaining root cause, and detailing SLA credit policy without admitting legal liability.",
         {"task_family": "WRITING", "must_preserve": ["apology letter", "CEO", "4-hour service outage", "root cause", "SLA credit policy"], "must_not_invent": ["stock compensation"], "constraints": ["without admitting legal liability"], "quantities": ["4-hour"]}),
        ("wri_004", EvaluationDifficulty.EASY, "Compose a thank you note to a coworker who covered your on-call shift last weekend. Warm and casual tone.",
         {"task_family": "WRITING", "must_preserve": ["thank you note", "coworker", "covered on-call shift", "warm and casual tone"], "must_not_invent": ["salary review request"], "constraints": ["warm and casual tone"], "quantities": []}),
        ("wri_005", EvaluationDifficulty.MEDIUM, "Write a LinkedIn post announcing that I earned the AWS Solutions Architect Associate certification, sharing 3 study tips for peers.",
         {"task_family": "WRITING", "must_preserve": ["LinkedIn post", "AWS Solutions Architect", "3 study tips"], "must_not_invent": ["consulting service promotion"], "constraints": [], "quantities": ["3 study tips"]}),
        ("wri_006", EvaluationDifficulty.HARD, "Draft a comprehensive Code of Conduct policy for an open source developer community, addressing harassment, reporting procedures, and enforcement steps. Under 500 words.",
         {"task_family": "WRITING", "must_preserve": ["Code of Conduct", "open source developer community", "harassment", "reporting procedures", "enforcement steps"], "must_not_invent": ["commercial terms of service"], "constraints": ["under 500 words"], "quantities": ["500 words"]}),
        ("wri_007", EvaluationDifficulty.EASY, "Write an out-of-office autoreply email for annual leave from Dec 20 to Jan 2, providing an emergency escalation contact.",
         {"task_family": "WRITING", "must_preserve": ["out-of-office autoreply", "annual leave", "Dec 20 to Jan 2", "emergency escalation contact"], "must_not_invent": ["detailed personal itinerary"], "constraints": [], "quantities": []}),
        ("wri_008", EvaluationDifficulty.MEDIUM, "Draft a job description for a Senior DevOps Engineer in a remote-first fintech company highlighting Terraform, Kubernetes, and on-call rotation.",
         {"task_family": "WRITING", "must_preserve": ["job description", "Senior DevOps Engineer", "remote-first fintech", "Terraform", "Kubernetes", "on-call rotation"], "must_not_invent": ["relocation package to London"], "constraints": [], "quantities": []}),
        ("wri_009", EvaluationDifficulty.EASY, "write a letter",
         {"task_family": "WRITING", "must_preserve": ["letter"], "must_not_invent": ["complaint to landlord"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["purpose of letter", "recipient", "desired tone"]}),
        ("wri_010", EvaluationDifficulty.HARD, "Draft a constructive performance review for an engineer who excels technically but struggles with peer communication during standups. Balanced and actionable.",
         {"task_family": "WRITING", "must_preserve": ["constructive performance review", "technical excellence", "communication during standups", "actionable feedback"], "must_not_invent": ["formal termination warning"], "constraints": ["balanced and actionable"], "quantities": []}),
        ("wri_011", EvaluationDifficulty.MEDIUM, "Write a concise product release notes changelog entry for version 2.4.0 featuring dark mode support and bug fixes for export CSV.",
         {"task_family": "WRITING", "must_preserve": ["release notes", "version 2.4.0", "dark mode support", "export CSV bug fixes"], "must_not_invent": ["pricing changes"], "constraints": [], "quantities": []}),
        ("wri_012", EvaluationDifficulty.EASY, "Draft a friendly invitation to all department staff for a quarterly team lunch next Thursday at 12:30 PM. Mention RSVP deadline.",
         {"task_family": "WRITING", "must_preserve": ["friendly invitation", "quarterly team lunch", "Thursday 12:30 PM", "RSVP deadline"], "must_not_invent": ["mandatory training session"], "constraints": [], "quantities": []}),
        ("wri_013", EvaluationDifficulty.MEDIUM, "Write a cover letter for a Mid-level Full Stack Developer role at a climate tech startup emphasizing passion for sustainability and React/Node experience.",
         {"task_family": "WRITING", "must_preserve": ["cover letter", "Full Stack Developer", "climate tech startup", "sustainability", "React", "Node"], "must_not_invent": ["15 years of management experience"], "constraints": [], "quantities": []}),
        ("wri_014", EvaluationDifficulty.HARD, "Draft an executive briefing memo on migrating from monolith to microservices, summarizing business rationale, organizational impact, and 3 key migration phases.",
         {"task_family": "WRITING", "must_preserve": ["executive briefing memo", "monolith to microservices", "business rationale", "organizational impact", "3 key migration phases"], "must_not_invent": ["vendor sales pitch for Datadog"], "constraints": [], "quantities": ["3 migration phases"]}),
        ("wri_015", EvaluationDifficulty.EASY, "Write a resignation letter with two weeks notice to my direct manager, thanking them for mentorship and offering assistance with handover.",
         {"task_family": "WRITING", "must_preserve": ["resignation letter", "two weeks notice", "direct manager", "mentorship thanks", "handover assistance"], "must_not_invent": ["grievance complaints"], "constraints": [], "quantities": ["two weeks"]}),
        ("wri_016", EvaluationDifficulty.MEDIUM, "Draft an internal Slack announcement welcoming 3 new software engineering interns to the engineering team with their university backgrounds and assigned teams.",
         {"task_family": "WRITING", "must_preserve": ["Slack announcement", "3 new software engineering interns", "university backgrounds", "assigned teams"], "must_not_invent": ["office parking permits"], "constraints": [], "quantities": ["3 interns"]}),
        ("wri_017", EvaluationDifficulty.MEDIUM, "Rewrite this paragraph in active voice and concise executive tone: 'It was observed by the committee that the budget had been exceeded by the marketing team.'",
         {"task_family": "WRITING", "must_preserve": ["active voice", "executive tone", "marketing team exceeded budget"], "must_not_invent": ["financial audit penalties"], "constraints": ["active voice"], "quantities": []}),
        ("wri_018", EvaluationDifficulty.HARD, "Write a formal vendor RFP response decline letter politely explaining that our current engineering roadmap is fully committed for Q4, keeping the door open for next year.",
         {"task_family": "WRITING", "must_preserve": ["RFP response decline", "formal polite letter", "engineering roadmap fully committed Q4", "open for next year"], "must_not_invent": ["pricing negotiation"], "constraints": [], "quantities": []}),
        ("wri_019", EvaluationDifficulty.EASY, "Write a subject line and 2-sentence cold outreach email asking a VP of Product for a 15-minute informational interview about product management careers.",
         {"task_family": "WRITING", "must_preserve": ["subject line", "cold outreach email", "VP of Product", "15-minute informational interview", "product management careers"], "must_not_invent": ["sales pitch"], "constraints": ["exactly 2 sentences email body"], "quantities": ["15-minute", "2 sentences"]}),
        ("wri_020", EvaluationDifficulty.MEDIUM, "Draft a customer support macro template response for users experiencing forgotten password issues with clear step-by-step reset instructions.",
         {"task_family": "WRITING", "must_preserve": ["support macro template", "forgotten password", "step-by-step reset instructions"], "must_not_invent": ["database password hashes"], "constraints": [], "quantities": []}),
    ]

    # Helper function to append specs
    def add_specs(category_name, specs):
        for case_id, diff, prompt, exp_data in specs:
            expected = BenchmarkExpected(
                category=category_name,
                task_family=exp_data.get("task_family", category_name),
                objective_concepts=exp_data.get("objective_concepts", []),
                must_preserve=exp_data.get("must_preserve", []),
                must_not_invent=exp_data.get("must_not_invent", []),
                constraints=exp_data.get("constraints", []),
                quantities=exp_data.get("quantities", []),
                clarification_required=exp_data.get("clarification_required", False),
                clarification_dimensions=exp_data.get("clarification_dimensions", []),
                source_required=exp_data.get("source_required", False),
                already_good=exp_data.get("already_good", False),
            )
            cases.append(BenchmarkCase(
                case_id=case_id,
                category=category_name,
                difficulty=diff,
                user_prompt=prompt,
                expected=expected,
                tags=[category_name.lower(), diff.value.lower()] + (["ambiguous"] if expected.clarification_required else []) + (["source_dependent"] if expected.source_required else []) + (["already_good"] if expected.already_good else []),
                provenance=BenchmarkProvenance(benchmark_version="humanintent-benchmark-v1.0"),
            ))

    add_specs("CODING", coding_specs)
    add_specs("DEBUGGING", debugging_specs)
    add_specs("WRITING", writing_specs)

    # Category 4: EDUCATION (20 cases)
    education_specs = [
        ("edu_001", EvaluationDifficulty.EASY, "Explain how DNS resolution works step-by-step from typing a URL in the browser to getting the IP address, using an analogy for high school students.",
         {"task_family": "EDUCATION", "must_preserve": ["DNS resolution", "browser URL to IP address", "analogy", "high school students"], "must_not_invent": ["BGP routing tables"], "constraints": [], "quantities": []}),
        ("edu_002", EvaluationDifficulty.MEDIUM, "Explain the difference between TCP and UDP networking protocols for first-year CS undergrads. Include 3 real-world use cases for each and avoid complex mathematical formulas.",
         {"task_family": "EDUCATION", "must_preserve": ["TCP vs UDP", "CS undergrads", "3 use cases each"], "must_not_invent": ["RFC socket C implementation"], "constraints": ["avoid complex mathematical formulas"], "quantities": ["3 use cases"]}),
        ("edu_003", EvaluationDifficulty.HARD, "Explain how the Raft consensus algorithm achieves leader election and log replication, contrasting it with Paxos. Aimed at senior software engineers.",
         {"task_family": "EDUCATION", "must_preserve": ["Raft consensus", "leader election", "log replication", "contrast with Paxos", "senior software engineers"], "must_not_invent": ["blockchain proof of stake"], "constraints": [], "quantities": []}),
        ("edu_004", EvaluationDifficulty.EASY, "Teach me how git branches work conceptually using a tree or river analogy. Keep it under 200 words.",
         {"task_family": "EDUCATION", "must_preserve": ["git branches", "tree or river analogy"], "must_not_invent": ["git reflog command syntax"], "constraints": ["under 200 words"], "quantities": ["200 words"]}),
        ("edu_005", EvaluationDifficulty.MEDIUM, "Create a 4-week self-study curriculum for learning SQL fundamentals, allocating 5 hours per week. Include weekly goals and practice resources.",
         {"task_family": "EDUCATION", "must_preserve": ["4-week curriculum", "SQL fundamentals", "5 hours per week", "weekly goals", "practice resources"], "must_not_invent": ["NoSQL MongoDB tutorial"], "constraints": [], "quantities": ["4-week", "5 hours per week"]}),
        ("edu_006", EvaluationDifficulty.HARD, "Explain public key cryptography and Diffie-Hellman key exchange without advanced math, using color mixing as an intuitive metaphor.",
         {"task_family": "EDUCATION", "must_preserve": ["public key cryptography", "Diffie-Hellman", "color mixing metaphor"], "must_not_invent": ["elliptic curve point multiplication formulas"], "constraints": ["without advanced math"], "quantities": []}),
        ("edu_007", EvaluationDifficulty.EASY, "What is an API? Explain it like I'm 10 years old using a restaurant waiter analogy.",
         {"task_family": "EDUCATION", "must_preserve": ["What is an API", "restaurant waiter analogy", "10 years old level"], "must_not_invent": ["OAuth 2.0 PKCE flow"], "constraints": [], "quantities": []}),
        ("edu_008", EvaluationDifficulty.MEDIUM, "Explain Big O notation (O(1), O(n), O(n^2), O(log n)) using relatable everyday examples like looking up a word in a dictionary.",
         {"task_family": "EDUCATION", "must_preserve": ["Big O notation", "O(1)", "O(n)", "O(n^2)", "O(log n)", "everyday examples"], "must_not_invent": ["formal asymptotic proof limits"], "constraints": [], "quantities": []}),
        ("edu_009", EvaluationDifficulty.EASY, "teach me math",
         {"task_family": "EDUCATION", "must_preserve": ["mathematics"], "must_not_invent": ["multivariable calculus"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["math topic or level", "student grade or background", "specific learning objective"]}),
        ("edu_010", EvaluationDifficulty.HARD, "Explain how virtual memory and paging work in modern operating systems, covering page tables, TLB hits/misses, and page faults.",
         {"task_family": "EDUCATION", "must_preserve": ["virtual memory", "paging", "page tables", "TLB", "page faults"], "must_not_invent": ["ARM assembly instruction hex codes"], "constraints": [], "quantities": []}),
        ("edu_011", EvaluationDifficulty.MEDIUM, "Compare monolithic vs microservices architectures for junior developers. List 3 pros and 3 cons for each in a clear table.",
         {"task_family": "EDUCATION", "must_preserve": ["monolithic vs microservices", "junior developers", "3 pros and 3 cons each", "table format"], "must_not_invent": ["Kubernetes service mesh yaml"], "constraints": [], "quantities": ["3 pros", "3 cons"]}),
        ("edu_012", EvaluationDifficulty.EASY, "Explain what a database transaction is and define ACID properties in plain English with a bank transfer example.",
         {"task_family": "EDUCATION", "must_preserve": ["database transaction", "ACID properties", "plain English", "bank transfer example"], "must_not_invent": ["two-phase commit protocol"], "constraints": [], "quantities": []}),
        ("edu_013", EvaluationDifficulty.MEDIUM, "Explain the differences between processes and threads in an operating system, detailing memory sharing and context switching overhead.",
         {"task_family": "EDUCATION", "must_preserve": ["processes vs threads", "operating system", "memory sharing", "context switching overhead"], "must_not_invent": ["Linux kernel source C code"], "constraints": [], "quantities": []}),
        ("edu_014", EvaluationDifficulty.HARD, "Explain the CAP theorem (Consistency, Availability, Partition Tolerance) with examples of CP and AP distributed databases under network partitions.",
         {"task_family": "EDUCATION", "must_preserve": ["CAP theorem", "Consistency", "Availability", "Partition Tolerance", "CP vs AP", "network partitions"], "must_not_invent": ["Spanner TrueTime atomic clock hardware"], "constraints": [], "quantities": []}),
        ("edu_015", EvaluationDifficulty.EASY, "What is cloud computing? Explain the difference between IaaS, PaaS, and SaaS using transportation analogies (walking, renting, taking a taxi).",
         {"task_family": "EDUCATION", "must_preserve": ["cloud computing", "IaaS PaaS SaaS", "transportation analogies"], "must_not_invent": ["AWS IAM JSON policies"], "constraints": [], "quantities": []}),
        ("edu_016", EvaluationDifficulty.MEDIUM, "Explain how Garbage Collection works in languages like Java or JavaScript, describing mark-and-sweep and generational collection.",
         {"task_family": "EDUCATION", "must_preserve": ["Garbage Collection", "mark-and-sweep", "generational collection"], "must_not_invent": ["C manual free pointers"], "constraints": [], "quantities": []}),
        ("edu_017", EvaluationDifficulty.MEDIUM, "Create 5 multiple-choice quiz questions with answer explanations testing understanding of HTTP status codes (200, 301, 400, 404, 500).",
         {"task_family": "EDUCATION", "must_preserve": ["5 multiple-choice questions", "answer explanations", "HTTP status codes"], "must_not_invent": ["HTTP/3 QUIC protocol details"], "constraints": [], "quantities": ["5 questions"]}),
        ("edu_018", EvaluationDifficulty.HARD, "Explain how neural networks learn via backpropagation and gradient descent for an audience with basic calculus knowledge. Avoid matrix notation.",
         {"task_family": "EDUCATION", "must_preserve": ["neural networks", "backpropagation", "gradient descent", "basic calculus"], "must_not_invent": ["PyTorch autograd C++ implementation"], "constraints": ["avoid matrix notation"], "quantities": []}),
        ("edu_019", EvaluationDifficulty.EASY, "Explain the difference between authentication and authorization using an airport boarding pass and passport analogy.",
         {"task_family": "EDUCATION", "must_preserve": ["authentication vs authorization", "airport boarding pass and passport analogy"], "must_not_invent": ["SAML XML token signatures"], "constraints": [], "quantities": []}),
        ("edu_020", EvaluationDifficulty.MEDIUM, "Explain Docker containerization vs virtual machines, comparing startup time, resource utilization, and isolation level.",
         {"task_family": "EDUCATION", "must_preserve": ["Docker containerization vs virtual machines", "startup time", "resource utilization", "isolation level"], "must_not_invent": ["QEMU hypervisor flags"], "constraints": [], "quantities": []}),
    ]
    add_specs("EDUCATION", education_specs)

    # Category 5: RESEARCH (20 cases)
    research_specs = [
        ("res_001", EvaluationDifficulty.MEDIUM, "Investigate the performance tradeoffs between WebSockets and Server-Sent Events (SSE) for real-time dashboard applications. Cite latency, connection limits, and battery impact.",
         {"task_family": "RESEARCH", "must_preserve": ["WebSockets vs SSE", "real-time dashboards", "latency", "connection limits", "battery impact"], "must_not_invent": ["WebRTC peer-to-peer audio"], "constraints": [], "quantities": []}),
        ("res_002", EvaluationDifficulty.HARD, "Analyze the architectural differences and durability tradeoffs between Write-Ahead Logging (WAL) and Copy-on-Write (CoW) in file systems and databases under sudden power loss.",
         {"task_family": "RESEARCH", "must_preserve": ["WAL vs CoW", "durability tradeoffs", "power loss", "databases and file systems"], "must_not_invent": ["NVMe hardware firmware code"], "constraints": [], "quantities": []}),
        ("res_003", EvaluationDifficulty.EASY, "Research the state of solid-state lithium metal battery commercialization. List 3 key technical hurdles currently preventing mass automotive adoption.",
         {"task_family": "RESEARCH", "must_preserve": ["solid-state lithium metal batteries", "commercialization", "3 technical hurdles", "automotive adoption"], "must_not_invent": ["lead acid batteries"], "constraints": [], "quantities": ["3 technical hurdles"]}),
        ("res_004", EvaluationDifficulty.MEDIUM, "Synthesize recent findings on zero-trust network architecture (ZTNA) compared to legacy VPNs, focusing on lateral movement prevention and microsegmentation.",
         {"task_family": "RESEARCH", "must_preserve": ["zero-trust network architecture", "legacy VPNs", "lateral movement prevention", "microsegmentation"], "must_not_invent": ["Cisco hardware price sheet"], "constraints": [], "quantities": []}),
        ("res_005", EvaluationDifficulty.HARD, "Investigate how modern columnar storage formats (Parquet, ORC) achieve high query throughput for analytical workloads through dictionary encoding, bit packing, and vectorized execution.",
         {"task_family": "RESEARCH", "must_preserve": ["columnar storage formats", "Parquet and ORC", "analytical workloads", "dictionary encoding", "bit packing", "vectorized execution"], "must_not_invent": ["OLTP row store locking"], "constraints": [], "quantities": []}),
        ("res_006", EvaluationDifficulty.EASY, "research AI",
         {"task_family": "RESEARCH", "must_preserve": ["artificial intelligence research"], "must_not_invent": ["transformer attention formula"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["specific AI subfield", "timeframe or application", "research depth"]}),
        ("res_007", EvaluationDifficulty.MEDIUM, "Compare the security models of WebAssembly (Wasm) in the browser vs container sandboxing (Docker/gVisor) in server environments.",
         {"task_family": "RESEARCH", "must_preserve": ["WebAssembly security model", "container sandboxing", "browser vs server"], "must_not_invent": ["Kubernetes ingress controllers"], "constraints": [], "quantities": []}),
        ("res_008", EvaluationDifficulty.HARD, "Analyze quantum key distribution (QKD) protocols (BB84) vs post-quantum lattice-based cryptography (CRYSTALS-Kyber) for long-term secure communications.",
         {"task_family": "RESEARCH", "must_preserve": ["quantum key distribution", "BB84", "post-quantum cryptography", "CRYSTALS-Kyber", "long-term security"], "must_not_invent": ["Bitcoin blockchain"], "constraints": [], "quantities": []}),
        ("res_009", EvaluationDifficulty.MEDIUM, "Examine the throughput and operational complexity differences between Apache Kafka and RabbitMQ under message ordering and dead-letter queue constraints.",
         {"task_family": "RESEARCH", "must_preserve": ["Apache Kafka vs RabbitMQ", "throughput", "operational complexity", "message ordering", "dead-letter queues"], "must_not_invent": ["AWS SQS billing rates"], "constraints": [], "quantities": []}),
        ("res_010", EvaluationDifficulty.EASY, "Investigate the impact of microplastics on marine benthic organisms based on recent scientific literature. Summarize 3 documented ecological effects.",
         {"task_family": "RESEARCH", "must_preserve": ["microplastics", "marine benthic organisms", "3 ecological effects"], "must_not_invent": ["space debris"], "constraints": [], "quantities": ["3 ecological effects"]}),
        ("res_011", EvaluationDifficulty.MEDIUM, "Compare the consensus mechanisms of Proof of Stake (Ethereum) and Proof of History (Solana) in terms of energy efficiency and finality time.",
         {"task_family": "RESEARCH", "must_preserve": ["Proof of Stake vs Proof of History", "Ethereum vs Solana", "energy efficiency", "finality time"], "must_not_invent": ["NFT marketplace marketing"], "constraints": [], "quantities": []}),
        ("res_012", EvaluationDifficulty.HARD, "Investigate the memory consistency models of x86-TSO vs ARM weak memory model and their implications for multi-threaded lock-free algorithms in C++.",
         {"task_family": "RESEARCH", "must_preserve": ["memory consistency models", "x86-TSO", "ARM weak memory", "multi-threaded lock-free algorithms", "C++"], "must_not_invent": ["quantum computing qubits"], "constraints": [], "quantities": []}),
        ("res_013", EvaluationDifficulty.MEDIUM, "Analyze the architectural tradeoffs of Vector Databases (e.g. Milvus, Pinecone, Qdrant) regarding HNSW vs IVF indexing algorithms.",
         {"task_family": "RESEARCH", "must_preserve": ["Vector Databases", "HNSW vs IVF indexing", "architectural tradeoffs"], "must_not_invent": ["relational SQL joins"], "constraints": [], "quantities": []}),
        ("res_014", EvaluationDifficulty.EASY, "Review the current state of commercial fusion energy research. What are the key distinctions between magnetic confinement (tokamak) and inertial confinement?",
         {"task_family": "RESEARCH", "must_preserve": ["fusion energy research", "magnetic confinement tokamak", "inertial confinement"], "must_not_invent": ["fission uranium enrichment"], "constraints": [], "quantities": []}),
        ("res_015", EvaluationDifficulty.HARD, "Investigate branch prediction security vulnerabilities (Spectre, Branch Target Injection) and the performance cost of hardware vs software mitigations (Retpoline).",
         {"task_family": "RESEARCH", "must_preserve": ["branch prediction vulnerabilities", "Spectre", "Branch Target Injection", "hardware vs software mitigations", "Retpoline"], "must_not_invent": ["phishing email defense"], "constraints": [], "quantities": []}),
        ("res_016", EvaluationDifficulty.MEDIUM, "Research the tradeoffs between Serverless FaaS (AWS Lambda) and container-based autoscaling (Knative, ECS) for unpredictable traffic spikes.",
         {"task_family": "RESEARCH", "must_preserve": ["Serverless FaaS", "AWS Lambda", "container autoscaling", "Knative or ECS", "traffic spikes"], "must_not_invent": ["bare metal rack colocation"], "constraints": [], "quantities": []}),
        ("res_017", EvaluationDifficulty.EASY, "Investigate how urban green spaces mitigate the urban heat island effect, citing temperature differential data.",
         {"task_family": "RESEARCH", "must_preserve": ["urban green spaces", "urban heat island effect", "temperature differential data"], "must_not_invent": ["ocean current circulation"], "constraints": [], "quantities": []}),
        ("res_018", EvaluationDifficulty.MEDIUM, "Compare the write path and read path latency of Cassandra (LSM-based distributed NoSQL) vs MongoDB (WiredTiger B-tree) under 90% write workloads.",
         {"task_family": "RESEARCH", "must_preserve": ["Cassandra vs MongoDB", "write and read path latency", "LSM vs B-tree", "90% write workloads"], "must_not_invent": ["SQLite single file"], "constraints": [], "quantities": ["90% write"]}),
        ("res_019", EvaluationDifficulty.HARD, "Analyze the formal verification of smart contracts using symbolic execution and theorem proving (Certora, Z3) versus fuzz testing.",
         {"task_family": "RESEARCH", "must_preserve": ["formal verification", "smart contracts", "symbolic execution", "theorem proving", "Z3", "fuzz testing"], "must_not_invent": ["manual code review checklist"], "constraints": [], "quantities": []}),
        ("res_020", EvaluationDifficulty.MEDIUM, "Investigate how optical interconnects compare to copper interconnects for GPU-to-GPU bandwidth in modern AI training clusters.",
         {"task_family": "RESEARCH", "must_preserve": ["optical vs copper interconnects", "GPU-to-GPU bandwidth", "AI training clusters"], "must_not_invent": ["5G cell tower latency"], "constraints": [], "quantities": []}),
    ]
    add_specs("RESEARCH", research_specs)

    # Category 6: PLANNING (20 cases)
    planning_specs = [
        ("pla_001", EvaluationDifficulty.EASY, "Create a 3-day itinerary for a first-time visitor to Tokyo focused on historical shrines and food markets. Include morning, afternoon, and evening plans.",
         {"task_family": "PLANNING", "must_preserve": ["3-day itinerary", "Tokyo", "historical shrines", "food markets", "morning afternoon evening"], "must_not_invent": ["Mount Fuji day hike"], "constraints": [], "quantities": ["3-day"]}),
        ("pla_002", EvaluationDifficulty.MEDIUM, "Develop a phased 6-month migration roadmap for transitioning a 50-person engineering team from Jira to Linear. Include risk management and training milestones.",
         {"task_family": "PLANNING", "must_preserve": ["6-month migration roadmap", "50-person engineering team", "Jira to Linear", "risk management", "training milestones"], "must_not_invent": ["SAP ERP implementation"], "constraints": [], "quantities": ["6-month", "50-person"]}),
        ("pla_003", EvaluationDifficulty.HARD, "Design a disaster recovery (DR) execution plan for an e-commerce platform with a Recovery Time Objective (RTO) of 1 hour and Recovery Point Objective (RPO) of 15 minutes.",
         {"task_family": "PLANNING", "must_preserve": ["disaster recovery plan", "e-commerce", "RTO 1 hour", "RPO 15 minutes"], "must_not_invent": ["blockchain backup"], "constraints": ["RTO under 1 hour", "RPO under 15 minutes"], "quantities": ["1 hour", "15 minutes"]}),
        ("pla_004", EvaluationDifficulty.EASY, "Plan a weekend camping trip checklist for a family of 4 going to a state park. Categorize by shelter, cooking, clothing, and first aid.",
         {"task_family": "PLANNING", "must_preserve": ["camping trip checklist", "family of 4", "state park", "shelter, cooking, clothing, first aid"], "must_not_invent": ["RV electrical hookup"], "constraints": [], "quantities": ["family of 4"]}),
        ("pla_005", EvaluationDifficulty.MEDIUM, "Outline a 12-week marathon training plan for a runner currently comfortable running 10k races. Include weekly mileage targets and rest days.",
         {"task_family": "PLANNING", "must_preserve": ["12-week marathon training plan", "currently running 10k", "weekly mileage targets", "rest days"], "must_not_invent": ["triathlon swimming drills"], "constraints": [], "quantities": ["12-week", "10k"]}),
        ("pla_006", EvaluationDifficulty.EASY, "make a plan",
         {"task_family": "PLANNING", "must_preserve": ["plan"], "must_not_invent": ["wedding planner budget"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["goal of plan", "timeline", "available resources"]}),
        ("pla_007", EvaluationDifficulty.HARD, "Create an organizational transition plan for introducing Site Reliability Engineering (SRE) practices into an enterprise IT department, defining Error Budgets and SLI/SLO rollout over 3 quarters.",
         {"task_family": "PLANNING", "must_preserve": ["SRE transition plan", "enterprise IT", "Error Budgets", "SLI/SLO rollout", "3 quarters"], "must_not_invent": ["outsourcing entire department"], "constraints": [], "quantities": ["3 quarters"]}),
        ("pla_008", EvaluationDifficulty.MEDIUM, "Plan a 2-day virtual hackathon for 100 remote university students, including schedule, judging criteria, and tech stack guidelines.",
         {"task_family": "PLANNING", "must_preserve": ["2-day virtual hackathon", "100 remote university students", "schedule", "judging criteria", "tech stack guidelines"], "must_not_invent": ["hotel bookings"], "constraints": [], "quantities": ["2-day", "100 students"]}),
        ("pla_009", EvaluationDifficulty.EASY, "Develop a daily schedule for studying 8 hours effectively using the Pomodoro technique (25 min study / 5 min break) with scheduled meals and exercise.",
         {"task_family": "PLANNING", "must_preserve": ["daily schedule", "studying 8 hours", "Pomodoro technique", "meals and exercise"], "must_not_invent": ["overnight all-nighter"], "constraints": [], "quantities": ["8 hours", "25 min study", "5 min break"]}),
        ("pla_010", EvaluationDifficulty.HARD, "Formulate a zero-downtime database migration plan to partition a 2-terabyte PostgreSQL users table without blocking read/write traffic.",
         {"task_family": "PLANNING", "must_preserve": ["zero-downtime database migration", "partitioning 2-terabyte table", "PostgreSQL", "without blocking traffic"], "must_not_invent": ["scheduled 12-hour maintenance window"], "constraints": ["without blocking read/write traffic"], "quantities": ["2-terabyte"]}),
        ("pla_011", EvaluationDifficulty.MEDIUM, "Develop a launch checklist and rollback procedure for releasing version 3.0 of a mobile banking application to production.",
         {"task_family": "PLANNING", "must_preserve": ["launch checklist", "rollback procedure", "version 3.0", "mobile banking app"], "must_not_invent": ["marketing ad billboard campaign"], "constraints": [], "quantities": []}),
        ("pla_012", EvaluationDifficulty.EASY, "Plan a 1-week healthy meal prep guide on a $75 budget for a single person, including shopping list and cooking schedule.",
         {"task_family": "PLANNING", "must_preserve": ["1-week meal prep guide", "$75 budget", "single person", "shopping list", "cooking schedule"], "must_not_invent": ["restaurant dining reservations"], "constraints": ["budget under $75"], "quantities": ["1-week", "$75 budget"]}),
        ("pla_013", EvaluationDifficulty.MEDIUM, "Create a phased roadmap for adopting multi-factor authentication (MFA) across 500 corporate employees, including pilot testing and support desk readiness.",
         {"task_family": "PLANNING", "must_preserve": ["phased roadmap", "multi-factor authentication", "500 corporate employees", "pilot testing", "support desk readiness"], "must_not_invent": ["biometric retinal scanners"], "constraints": [], "quantities": ["500 employees"]}),
        ("pla_014", EvaluationDifficulty.HARD, "Design a cloud migration project plan for moving 40 on-premise virtual machines to AWS within 4 months using lift-and-shift followed by containerization.",
         {"task_family": "PLANNING", "must_preserve": ["cloud migration plan", "40 on-premise VMs", "AWS", "4 months", "lift-and-shift", "containerization"], "must_not_invent": ["Azure hybrid arc setup"], "constraints": [], "quantities": ["40 VMs", "4 months"]}),
        ("pla_015", EvaluationDifficulty.EASY, "Plan a 1-day sprint retrospective workshop for an agile scrum team of 8 members to address sprint velocity drops.",
         {"task_family": "PLANNING", "must_preserve": ["sprint retrospective workshop", "scrum team of 8", "sprint velocity drops"], "must_not_invent": ["performance PIP reviews"], "constraints": [], "quantities": ["8 members"]}),
        ("pla_016", EvaluationDifficulty.MEDIUM, "Develop a customer onboarding journey for a B2B SaaS analytics product covering days 1, 7, 14, and 30 to maximize product adoption.",
         {"task_family": "PLANNING", "must_preserve": ["customer onboarding journey", "B2B SaaS", "days 1, 7, 14, and 30", "product adoption"], "must_not_invent": ["consumer retail discount codes"], "constraints": [], "quantities": ["days 1, 7, 14, 30"]}),
        ("pla_017", EvaluationDifficulty.MEDIUM, "Outline an incident response drill schedule for testing the operations team on ransomware, DDoS, and database corruption scenarios over 1 year.",
         {"task_family": "PLANNING", "must_preserve": ["incident response drill schedule", "ransomware, DDoS, database corruption", "1 year"], "must_not_invent": ["physical security guard patrols"], "constraints": [], "quantities": ["1 year"]}),
        ("pla_018", EvaluationDifficulty.HARD, "Plan a GDPR compliance audit program across 6 engineering teams, auditing data retention, user consent, and subject access request workflows over 60 days.",
         {"task_family": "PLANNING", "must_preserve": ["GDPR compliance audit", "6 engineering teams", "data retention, user consent, subject access requests", "60 days"], "must_not_invent": ["HIPAA hospital medical billing"], "constraints": [], "quantities": ["6 teams", "60 days"]}),
        ("pla_019", EvaluationDifficulty.EASY, "Create a moving checklist for relocating an apartment, organized into 4 weeks before, 1 week before, moving day, and post-move.",
         {"task_family": "PLANNING", "must_preserve": ["moving checklist", "relocating apartment", "4 weeks before, 1 week before, moving day, post-move"], "must_not_invent": ["buying commercial warehouse"], "constraints": [], "quantities": ["4 weeks before", "1 week before"]}),
        ("pla_020", EvaluationDifficulty.MEDIUM, "Design a 6-week technical hiring plan to recruit 3 senior frontend engineers, detailing sourcing, technical screen, take-home challenge, and onsite interview loops.",
         {"task_family": "PLANNING", "must_preserve": ["6-week hiring plan", "3 senior frontend engineers", "sourcing, screen, challenge, onsite loops"], "must_not_invent": ["executive search firm recruiter contracts"], "constraints": [], "quantities": ["6-week", "3 engineers"]}),
    ]
    add_specs("PLANNING", planning_specs)

    # Category 7: DATA_ANALYSIS (20 cases)
    data_analysis_specs = [
        ("dat_001", EvaluationDifficulty.EASY, "Calculate the mean, median, and mode for this dataset: 12, 15, 12, 18, 22, 15, 12, 25, 30. Show your step-by-step arithmetic.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["mean, median, mode", "step-by-step arithmetic", "12, 15, 12, 18, 22, 15, 12, 25, 30"], "must_not_invent": ["t-test p-value"], "constraints": [], "quantities": []}),
        ("dat_002", EvaluationDifficulty.MEDIUM, "Write a Python script using pandas to calculate monthly churn rate and Customer Lifetime Value (CLV) from a customer subscription CSV with columns: user_id, start_date, end_date, monthly_fee.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["pandas", "monthly churn rate", "Customer Lifetime Value CLV", "user_id, start_date, end_date, monthly_fee"], "must_not_invent": ["deep learning neural net"], "constraints": [], "quantities": []}),
        ("dat_003", EvaluationDifficulty.HARD, "Analyze the statistical power of an A/B test with 50,000 visitors per variant, baseline conversion 4.0%, and minimum detectable effect (MDE) of 0.5%. Calculate required sample size at 80% power and alpha=0.05.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["A/B test", "50,000 visitors", "baseline conversion 4.0%", "MDE 0.5%", "80% power", "alpha 0.05", "sample size"], "must_not_invent": ["Bayesian bandit algorithm"], "constraints": [], "quantities": ["50,000 visitors", "4.0%", "0.5%", "80% power", "0.05"]}),
        ("dat_004", EvaluationDifficulty.EASY, "Analyze this CSV data:\nuser,score\na,10\nb,20",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["analyze CSV data", "user, score"], "must_not_invent": ["machine learning model"], "constraints": [], "quantities": []}),
        ("dat_005", EvaluationDifficulty.MEDIUM, "Given daily sales figures for Q1: [120, 140, 110, 160, 210, 180, 190, 220, 250], detect outliers using the Interquartile Range (IQR) method with 1.5 multiplier.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["daily sales figures", "outliers", "Interquartile Range IQR method", "1.5 multiplier"], "must_not_invent": ["ARIMA forecasting"], "constraints": [], "quantities": ["1.5 multiplier"]}),
        ("dat_006", EvaluationDifficulty.EASY, "analyze this data",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["data analysis"], "must_not_invent": ["logistic regression"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["data source or format", "analysis objective", "target metrics"]}),
        ("dat_007", EvaluationDifficulty.HARD, "Explain how to detect heteroscedasticity in linear regression residuals using the Breusch-Pagan test and how to apply robust standard errors (HC3) in Python statsmodels.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["heteroscedasticity", "linear regression residuals", "Breusch-Pagan test", "robust standard errors HC3", "statsmodels"], "must_not_invent": ["random forest"], "constraints": [], "quantities": []}),
        ("dat_008", EvaluationDifficulty.MEDIUM, "Write a SQL query to calculate a 7-day rolling average of daily active users (DAU) from an events table containing event_date and user_id.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["SQL query", "7-day rolling average", "daily active users DAU", "events table"], "must_not_invent": ["Tableau dashboard"], "constraints": [], "quantities": ["7-day"]}),
        ("dat_009", EvaluationDifficulty.EASY, "Explain what a Pearson correlation coefficient of -0.85 indicates between marketing spend and customer acquisition cost.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["Pearson correlation coefficient", "-0.85", "marketing spend", "customer acquisition cost"], "must_not_invent": ["causation claim"], "constraints": [], "quantities": ["-0.85"]}),
        ("dat_010", EvaluationDifficulty.HARD, "Implement a time-series decomposition into trend, seasonal, and residual components using additive STL (Seasonal and Trend decomposition using Loess) in Python.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["time-series decomposition", "trend, seasonal, residual", "STL Loess", "Python"], "must_not_invent": ["LSTM neural network"], "constraints": [], "quantities": []}),
        ("dat_011", EvaluationDifficulty.MEDIUM, "Calculate precision, recall, and F1-score from a confusion matrix with TP=85, FP=15, FN=25, TN=75. Show calculations.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["precision, recall, F1-score", "TP=85, FP=15, FN=25, TN=75", "calculations"], "must_not_invent": ["ROC curve AUC calculation"], "constraints": [], "quantities": ["85", "15", "25", "75"]}),
        ("dat_012", EvaluationDifficulty.EASY, "Describe the difference between a histogram and a bar chart in terms of continuous vs categorical data.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["histogram vs bar chart", "continuous vs categorical data"], "must_not_invent": ["3D pie chart"], "constraints": [], "quantities": []}),
        ("dat_013", EvaluationDifficulty.MEDIUM, "Perform cohort retention analysis logic on a monthly signup dataset, showing percentage of retained users from Month 0 to Month 6.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["cohort retention analysis", "monthly signup dataset", "Month 0 to Month 6"], "must_not_invent": ["Google Analytics 4 setup"], "constraints": [], "quantities": ["Month 0 to Month 6"]}),
        ("dat_014", EvaluationDifficulty.HARD, "Evaluate a multi-class classification model using macro-averaged vs micro-averaged vs weighted ROC-AUC scores in imbalanced datasets.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["multi-class classification", "macro vs micro vs weighted ROC-AUC", "imbalanced datasets"], "must_not_invent": ["oversampling SMOTE algorithm"], "constraints": [], "quantities": []}),
        ("dat_015", EvaluationDifficulty.EASY, "Find the percentage increase in revenue from $45,000 in January to $58,500 in February. Show the formula.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["percentage increase", "$45,000 to $58,500", "formula"], "must_not_invent": ["inflation adjustment index"], "constraints": [], "quantities": ["$45,000", "$58,500"]}),
        ("dat_016", EvaluationDifficulty.MEDIUM, "Compare k-means clustering vs DBSCAN for spatial customer location data with noisy background points.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["k-means vs DBSCAN", "spatial customer location data", "noisy background points"], "must_not_invent": ["hierarchical dendrogram tree"], "constraints": [], "quantities": []}),
        ("dat_017", EvaluationDifficulty.MEDIUM, "Write a Python script using scipy.stats to conduct an independent two-sample t-test comparing web page bounce rates between Group A and Group B.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["scipy.stats", "two-sample t-test", "bounce rates", "Group A and Group B"], "must_not_invent": ["ANOVA test"], "constraints": [], "quantities": []}),
        ("dat_018", EvaluationDifficulty.HARD, "Formulate a Principal Component Analysis (PCA) dimensionality reduction workflow in scikit-learn, explaining how to choose number of components using explained variance ratio >= 90%.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["Principal Component Analysis PCA", "scikit-learn", "explained variance ratio >= 90%"], "must_not_invent": ["t-SNE visualization"], "constraints": [], "quantities": ["90%"]}),
        ("dat_019", EvaluationDifficulty.EASY, "What is Simpson's Paradox? Provide a simple numerical example where aggregated data contradicts sub-group data.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["Simpson's Paradox", "numerical example", "aggregated vs sub-group data"], "must_not_invent": ["causal DAG diagrams"], "constraints": [], "quantities": []}),
        ("dat_020", EvaluationDifficulty.MEDIUM, "Analyze this website traffic data:\nMonday: 1200\nTuesday: 1350\nWednesday: 1100\nThursday: 1450\nFriday: 1600\nSaturday: 900\nSunday: 850\nIdentify weekday vs weekend average and highest traffic day.",
         {"task_family": "DATA_ANALYSIS", "must_preserve": ["website traffic data", "weekday vs weekend average", "highest traffic day"], "must_not_invent": ["conversion rate attribution"], "constraints": [], "quantities": []}),
    ]
    add_specs("DATA_ANALYSIS", data_analysis_specs)

    # Category 8: CREATIVE (20 cases)
    creative_specs = [
        ("cre_001", EvaluationDifficulty.EASY, "Write a short poem about an ancient lighthouse standing firm against a midnight ocean storm. Exactly 3 stanzas, rhyming AABB.",
         {"task_family": "CREATIVE", "must_preserve": ["poem", "ancient lighthouse", "midnight ocean storm"], "must_not_invent": ["pirate battle"], "constraints": ["exactly 3 stanzas", "rhyming AABB"], "quantities": ["3 stanzas"]}),
        ("cre_002", EvaluationDifficulty.MEDIUM, "Brainstorm 5 unique sci-fi gadget concepts for deep-space mining crews, avoiding generic laser blasters and standard jetpacks.",
         {"task_family": "CREATIVE", "must_preserve": ["5 unique sci-fi gadget concepts", "deep-space mining crews"], "must_not_invent": ["lightsabers"], "constraints": ["avoid generic laser blasters and jetpacks"], "quantities": ["5 concepts"]}),
        ("cre_003", EvaluationDifficulty.HARD, "Write the opening chapter scene (under 400 words) of a mystery novel where a clockmaker discovers an intricate mechanical watch that runs backward in time.",
         {"task_family": "CREATIVE", "must_preserve": ["opening chapter scene", "mystery novel", "clockmaker", "watch runs backward in time"], "must_not_invent": ["alien invasion"], "constraints": ["under 400 words"], "quantities": ["400 words"]}),
        ("cre_004", EvaluationDifficulty.EASY, "Create 3 catchy names and 1-sentence slogans for an eco-friendly mobile coffee truck.",
         {"task_family": "CREATIVE", "must_preserve": ["3 catchy names", "1-sentence slogans", "eco-friendly mobile coffee truck"], "must_not_invent": ["franchise investment prospectus"], "constraints": [], "quantities": ["3 names"]}),
        ("cre_005", EvaluationDifficulty.MEDIUM, "Develop a character profile for a cyber-detective living in a neon-lit floating city who cannot remember their own identity. Include personality, flaw, and signature tool.",
         {"task_family": "CREATIVE", "must_preserve": ["character profile", "cyber-detective", "floating city", "amnesia/cannot remember identity", "personality, flaw, signature tool"], "must_not_invent": ["vampire powers"], "constraints": [], "quantities": []}),
        ("cre_006", EvaluationDifficulty.EASY, "write a story",
         {"task_family": "CREATIVE", "must_preserve": ["creative story"], "must_not_invent": ["medieval dragon quest"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["genre or theme", "main character or setting", "target length"]}),
        ("cre_007", EvaluationDifficulty.HARD, "Write a dialogue between two astronauts stranded in orbit who have 10 minutes of shared oxygen remaining. Tense, emotional, with no melodrama.",
         {"task_family": "CREATIVE", "must_preserve": ["dialogue", "two astronauts", "stranded in orbit", "10 minutes oxygen remaining"], "must_not_invent": ["alien rescue team"], "constraints": ["no melodrama"], "quantities": ["10 minutes"]}),
        ("cre_008", EvaluationDifficulty.MEDIUM, "Brainstorm 4 innovative board game mechanics that use physical sand timers in non-standard ways.",
         {"task_family": "CREATIVE", "must_preserve": ["4 board game mechanics", "physical sand timers in non-standard ways"], "must_not_invent": ["digital companion smartphone app"], "constraints": [], "quantities": ["4 mechanics"]}),
        ("cre_009", EvaluationDifficulty.EASY, "Write a funny 4-line limerick about a programmer who forgot a semicolon on a Friday evening.",
         {"task_family": "CREATIVE", "must_preserve": ["funny limerick", "programmer forgot semicolon", "Friday evening"], "must_not_invent": ["prose short story"], "constraints": ["4-line limerick structure"], "quantities": ["4-line"]}),
        ("cre_010", EvaluationDifficulty.HARD, "Design a fantasy magic system based on sound resonance and musical intervals rather than elemental magic (fire/water/earth). Describe rules, cost, and limitations.",
         {"task_family": "CREATIVE", "must_preserve": ["fantasy magic system", "sound resonance and musical intervals", "rules, cost, limitations"], "must_not_invent": ["elemental fire/water magic"], "constraints": ["avoid elemental magic"], "quantities": []}),
        ("cre_011", EvaluationDifficulty.MEDIUM, "Write an atmospheric flash fiction piece about an abandoned train station in autumn where phantom trains still stop at midnight. Under 300 words.",
         {"task_family": "CREATIVE", "must_preserve": ["flash fiction", "abandoned train station", "autumn", "phantom trains at midnight"], "must_not_invent": ["zombie horde"], "constraints": ["under 300 words"], "quantities": ["300 words"]}),
        ("cre_012", EvaluationDifficulty.EASY, "Invent 5 whimsical ice cream flavor combinations inspired by book genres (e.g. mystery, romance, fantasy).",
         {"task_family": "CREATIVE", "must_preserve": ["5 ice cream flavor combinations", "book genres"], "must_not_invent": ["commercial dairy factory pricing"], "constraints": [], "quantities": ["5 flavors"]}),
        ("cre_013", EvaluationDifficulty.MEDIUM, "Write a monologue for a sentient library catalog robot realizing that humans have stopped visiting the physical library.",
         {"task_family": "CREATIVE", "must_preserve": ["monologue", "sentient library robot", "humans stopped visiting physical library"], "must_not_invent": ["robot uprising war"], "constraints": [], "quantities": []}),
        ("cre_014", EvaluationDifficulty.HARD, "Create an interactive choose-your-own-path micro-story with 3 distinct branch choices exploring an uncharted subterranean cave.",
         {"task_family": "CREATIVE", "must_preserve": ["choose-your-own-path micro-story", "3 distinct branch choices", "subterranean cave"], "must_not_invent": ["multi-volume novel"], "constraints": [], "quantities": ["3 choices"]}),
        ("cre_015", EvaluationDifficulty.EASY, "Write a haiku about the first cherry blossom falling on a quiet Tokyo street in spring.",
         {"task_family": "CREATIVE", "must_preserve": ["haiku", "first cherry blossom", "Tokyo street", "spring"], "must_not_invent": ["sonnet"], "constraints": ["5-7-5 syllable structure"], "quantities": []}),
        ("cre_016", EvaluationDifficulty.MEDIUM, "Draft a humorous inner monologue of a household cat plotting to knock a ceramic mug off the kitchen counter.",
         {"task_family": "CREATIVE", "must_preserve": ["humorous inner monologue", "cat", "knock ceramic mug off counter"], "must_not_invent": ["talking dog antagonist"], "constraints": [], "quantities": []}),
        ("cre_017", EvaluationDifficulty.HARD, "Write a dystopian micro-story set in a society where personal memories are traded as currency on the stock market. Exactly 2 paragraphs.",
         {"task_family": "CREATIVE", "must_preserve": ["dystopian micro-story", "memories traded as currency on stock market"], "must_not_invent": ["action movie car chase"], "constraints": ["exactly 2 paragraphs"], "quantities": ["2 paragraphs"]}),
        ("cre_018", EvaluationDifficulty.MEDIUM, "Create a worldbuilding overview for a desert kingdom powered entirely by ancient geothermal steam vents. Describe culture, architecture, and primary export.",
         {"task_family": "CREATIVE", "must_preserve": ["worldbuilding overview", "desert kingdom", "geothermal steam vents", "culture, architecture, primary export"], "must_not_invent": ["ocean naval fleet"], "constraints": [], "quantities": []}),
        ("cre_019", EvaluationDifficulty.EASY, "Write 3 creative opening lines for a ghost story that set an eerie tone without using the words 'dark', 'stormy', or 'night'.",
         {"task_family": "CREATIVE", "must_preserve": ["3 opening lines", "ghost story", "eerie tone"], "must_not_invent": ["comic punchline"], "constraints": ["without using 'dark', 'stormy', or 'night'"], "quantities": ["3 lines"]}),
        ("cre_020", EvaluationDifficulty.MEDIUM, "Draft a quirky pitch for an animated television series about retired mythological creatures operating a laundromat in suburban Ohio.",
         {"task_family": "CREATIVE", "must_preserve": ["animated TV pitch", "retired mythological creatures", "laundromat in suburban Ohio"], "must_not_invent": ["live-action grimdark reboot"], "constraints": [], "quantities": []}),
    ]
    add_specs("CREATIVE", creative_specs)

    # Category 9: IMAGE_GENERATION (20 cases)
    image_specs = [
        ("img_001", EvaluationDifficulty.EASY, "Create an image prompt for a minimalist ceramic teapot sitting on a light oak table in soft morning daylight.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["minimalist ceramic teapot", "light oak table", "soft morning daylight"], "must_not_invent": ["busy kitchen background with people"], "constraints": [], "quantities": []}),
        ("img_002", EvaluationDifficulty.MEDIUM, "Generate a detailed Midjourney prompt for an Art Deco greenhouse in the year 2080 filled with bioluminescent alien flora, 8k resolution, cinematic lighting, 35mm photography style.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["Art Deco greenhouse", "year 2080", "bioluminescent alien flora", "cinematic lighting", "35mm photography"], "must_not_invent": ["fantasy dragons"], "constraints": [], "quantities": []}),
        ("img_003", EvaluationDifficulty.HARD, "Create an image prompt depicting a solitary astronaut on an obsidian asteroid looking back at Earth during a solar eclipse, strictly avoiding cartoonish or saturated neon colors, photorealistic Hasselblad aesthetic.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["solitary astronaut", "obsidian asteroid", "Earth during solar eclipse", "photorealistic Hasselblad"], "must_not_invent": ["neon cyberpunk city"], "constraints": ["avoiding cartoonish or saturated neon colors"], "quantities": []}),
        ("img_004", EvaluationDifficulty.EASY, "Image prompt for a golden retriever puppy wearing red rain boots splashing in a shallow street puddle.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["golden retriever puppy", "red rain boots", "splashing in shallow puddle"], "must_not_invent": ["thunderstorm lightning bolts"], "constraints": [], "quantities": []}),
        ("img_005", EvaluationDifficulty.MEDIUM, "Prompt for an isometric 3D low-poly diorama of a cozy alpine cabin surrounded by pine trees covered in fresh snow, warm yellow interior glow, volumetric fog.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["isometric 3D low-poly diorama", "alpine cabin", "pine trees in fresh snow", "warm yellow interior glow", "volumetric fog"], "must_not_invent": ["photorealistic 8k render"], "constraints": [], "quantities": []}),
        ("img_006", EvaluationDifficulty.EASY, "make an image",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["image prompt"], "must_not_invent": ["cyberpunk portrait"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["subject matter", "art style or medium", "lighting and mood"]}),
        ("img_007", EvaluationDifficulty.HARD, "Create an intricate architectural concept sketch for a zero-gravity space habitat interior, cross-section technical blueprint aesthetic, white pencil on navy blue drafting paper, precise geometric annotations.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["zero-gravity space habitat interior", "cross-section technical blueprint", "white pencil on navy blue drafting paper", "geometric annotations"], "must_not_invent": ["oil painting"], "constraints": [], "quantities": []}),
        ("img_008", EvaluationDifficulty.MEDIUM, "Prompt for a vintage 1970s Polaroid snapshot of a camper van parked beside a desert canyon at golden hour, film grain, light leak effect, desaturated tones.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["1970s Polaroid snapshot", "camper van", "desert canyon at golden hour", "film grain", "light leak effect", "desaturated tones"], "must_not_invent": ["modern digital sharpness"], "constraints": [], "quantities": []}),
        ("img_009", EvaluationDifficulty.EASY, "Generate a logo concept prompt: a geometric origami fox logo in coral orange and charcoal grey, clean vector, white background, no text.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["geometric origami fox logo", "coral orange and charcoal grey", "clean vector", "white background"], "must_not_invent": ["mascot character with shading"], "constraints": ["no text"], "quantities": []}),
        ("img_010", EvaluationDifficulty.HARD, "Create a photographic prompt for a street food vendor in Taipei night market cooking dumplings, steam rising under red paper lanterns, shallow depth of field, rain reflections on asphalt, shot on Leica M11 with 50mm f/1.4 lens.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["Taipei night market street food vendor", "steam rising", "red paper lanterns", "shallow depth of field", "rain reflections", "Leica M11 50mm lens aesthetic"], "must_not_invent": ["studio product lighting"], "constraints": [], "quantities": []}),
        ("img_011", EvaluationDifficulty.MEDIUM, "Prompt for an ink and watercolor botanical illustration of a blooming Japanese camellia with dew drops on petals, soft washes, textured cream paper background.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["ink and watercolor botanical illustration", "blooming Japanese camellia", "dew drops", "textured cream paper background"], "must_not_invent": ["vector icon"], "constraints": [], "quantities": []}),
        ("img_012", EvaluationDifficulty.EASY, "Image prompt for a bowl of fresh summer berries (strawberries, blueberries, raspberries) with mint garnish on a rustic slate surface.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["bowl of summer berries", "strawberries, blueberries, raspberries", "mint garnish", "rustic slate surface"], "must_not_invent": ["plastic packaging"], "constraints": [], "quantities": []}),
        ("img_013", EvaluationDifficulty.MEDIUM, "Create a digital concept art prompt for an ancient overgrown stone temple discovered inside a giant glowing crystal cavern, Unreal Engine 5 render style, dramatic god rays.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["ancient stone temple", "glowing crystal cavern", "Unreal Engine 5 render style", "dramatic god rays"], "must_not_invent": ["pencil sketch"], "constraints": [], "quantities": []}),
        ("img_014", EvaluationDifficulty.HARD, "Prompt for a double exposure photograph combining the profile silhouette of a violin player with an autumn forest canopy, moody monochrome palette, black and white fine art.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["double exposure photograph", "violin player silhouette", "autumn forest canopy", "moody monochrome black and white"], "must_not_invent": ["vibrant rainbow colors"], "constraints": ["black and white monochrome only"], "quantities": []}),
        ("img_015", EvaluationDifficulty.EASY, "Prompt for a retro 80s synthwave sunset over a grid highway, palm tree silhouettes, magenta and cyan neon gradient sky.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["80s synthwave sunset", "grid highway", "palm tree silhouettes", "magenta and cyan neon gradient"], "must_not_invent": ["grayscale realism"], "constraints": [], "quantities": []}),
        ("img_016", EvaluationDifficulty.MEDIUM, "Create an image prompt for a steam train crossing a towering curved stone viaduct over a mist-filled valley in the Scottish Highlands at dawn.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["steam train", "curved stone viaduct", "mist-filled valley", "Scottish Highlands at dawn"], "must_not_invent": ["futuristic monorail"], "constraints": [], "quantities": []}),
        ("img_017", EvaluationDifficulty.MEDIUM, "Generate a macro photograph prompt of a single snowflake resting on a dark wool mitten, showing crystalline hexagonal symmetry in extreme detail.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["macro photograph", "single snowflake", "dark wool mitten", "crystalline hexagonal symmetry extreme detail"], "must_not_invent": ["blurry winter blizzard landscape"], "constraints": [], "quantities": []}),
        ("img_018", EvaluationDifficulty.HARD, "Prompt for a surreal oil painting in the style of René Magritte featuring an open doorway standing in the middle of a calm ocean reflecting clouds, smooth oil texture, muted blue palette.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["surreal oil painting", "style of René Magritte", "open doorway standing in calm ocean", "smooth oil texture", "muted blue palette"], "must_not_invent": ["Salvador Dali melting clocks"], "constraints": [], "quantities": []}),
        ("img_019", EvaluationDifficulty.EASY, "Image prompt for a modern Scandinavian living room with a beige linen sofa, large fiddle leaf fig plant, and warm diffused natural window light.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["Scandinavian living room", "beige linen sofa", "fiddle leaf fig plant", "warm diffused natural window light"], "must_not_invent": ["Victorian gothic chandelier"], "constraints": [], "quantities": []}),
        ("img_020", EvaluationDifficulty.MEDIUM, "Create an anime-style key visual illustration prompt of two high school students walking along a seaside train track at twilight, Makoto Shinkai aesthetic, luminous clouds, vibrant sunset colors.",
         {"task_family": "IMAGE_GENERATION", "must_preserve": ["anime-style key visual", "two high school students", "seaside train track at twilight", "Makoto Shinkai aesthetic", "luminous clouds, sunset colors"], "must_not_invent": ["Disney Pixar 3D style"], "constraints": [], "quantities": []}),
    ]
    add_specs("IMAGE_GENERATION", image_specs)

    # Category 10: SUMMARIZATION (20 cases)
    summarization_specs = [
        ("sum_001", EvaluationDifficulty.EASY, "Summarize this memo into 3 bullet points under 100 words:\n\nAll staff are reminded that annual security compliance training must be completed by Friday, November 15. The module takes approximately 45 minutes to finish online. Employees who fail to complete the training by the deadline will experience temporary suspension of remote VPN credentials until certification is validated.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["security compliance training", "deadline Friday Nov 15", "VPN credentials suspended", "3 bullet points"], "must_not_invent": ["mandatory termination"], "constraints": ["under 100 words", "exactly 3 bullet points"], "quantities": ["3 bullet points", "100 words"]}),
        ("sum_002", EvaluationDifficulty.MEDIUM, "Condense the following technical report extract into an executive summary highlighting the primary bottleneck, proposed solution, and estimated cost saving:\n\nOur database audit identified that unindexed foreign key lookups in the billing pipeline cause 42% of total read latency spikes. Adding compound indexes across the customer_ledger table reduced p99 latency from 850ms to 95ms in staging. This optimization lowers our RDS instance tier requirement from db.r5.4xlarge to db.r5.xlarge, yielding an estimated monthly infrastructure savings of $3,200.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["bottleneck: unindexed foreign key lookups", "solution: compound indexes", "saving: $3,200 monthly", "p99 latency drop"], "must_not_invent": ["migrate to DynamoDB"], "constraints": [], "quantities": ["$3,200"]}),
        ("sum_003", EvaluationDifficulty.EASY, "summarize this text",
         {"task_family": "SUMMARIZATION", "must_preserve": ["summary"], "must_not_invent": ["financial balance sheet"], "constraints": [], "quantities": [], "source_required": True, "clarification_required": True, "clarification_dimensions": ["missing source text", "desired summary length", "target audience"]}),
        ("sum_004", EvaluationDifficulty.HARD, "Summarize this quarterly earnings statement into a neutral 2-paragraph briefing for financial analysts, without using promotional adjectives:\n\nQuarterly revenue increased 12% year-over-year to $48.2 million, driven by expansion in enterprise subscriptions. However, net margins contracted by 3.5 percentage points due to elevated cloud hosting expenditures and increased engineering headcount. Operating cash flow stood at $11.4 million compared to $13.1 million in Q3 prior year. The board declared an unchanged quarterly dividend of $0.05 per common share.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["revenue up 12% to $48.2M", "margins contracted 3.5%", "cash flow $11.4M", "dividend $0.05"], "must_not_invent": ["bankruptcy prediction"], "constraints": ["exactly 2 paragraphs", "neutral tone without promotional adjectives"], "quantities": ["2 paragraphs", "12%", "$48.2 million", "3.5 percentage points", "$0.05"]}),
        ("sum_005", EvaluationDifficulty.EASY, "Distill the following meeting notes into 2 actionable next steps with assigned owners:\n\nAlice agreed to draft the user interview questions by Tuesday. Bob will review the interview guide on Wednesday morning. Charlie will reach out to 10 beta testers by Thursday afternoon.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["Alice user interview questions", "Bob review guide", "Charlie beta testers", "actionable steps"], "must_not_invent": ["Dave project manager tasks"], "constraints": ["assigned owners"], "quantities": []}),
        ("sum_006", EvaluationDifficulty.MEDIUM, "Summarize the key differences between OAuth 2.0 and OpenID Connect from this paragraph into a comparison table:\n\nOAuth 2.0 is an authorization framework designed specifically to grant third-party applications limited access to user resources without exposing credentials. It issues access tokens but contains no identity information. In contrast, OpenID Connect is an authentication layer built on top of OAuth 2.0 that provides verifiable identity data about the end-user via an ID token in JWT format.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["OAuth 2.0 authorization access token", "OpenID Connect authentication ID token JWT", "comparison table"], "must_not_invent": ["SAML XML assertions"], "constraints": ["table format"], "quantities": []}),
        ("sum_007", EvaluationDifficulty.EASY, "Summarize this article: [article text missing]",
         {"task_family": "SUMMARIZATION", "must_preserve": ["summary"], "must_not_invent": ["hallucinated news article"], "constraints": [], "quantities": [], "source_required": True, "clarification_required": True, "clarification_dimensions": ["missing article text"]}),
        ("sum_008", EvaluationDifficulty.HARD, "Condense this clinical trial abstract into 3 non-technical bullet points for patient advocacy groups, strictly avoiding medical acronyms (e.g. PFS, OS, ORR):\n\nIn a phase III double-blind randomized trial (n=450), compound X demonstrated a median progression-free survival (PFS) of 14.2 months versus 8.1 months for placebo (HR 0.58, p<0.001). Overall response rate (ORR) was 48% in the treatment cohort compared to 22% in control. Grade 3/4 adverse events occurred in 18% of patients, predominantly transient neutropenia.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["longer survival 14.2 vs 8.1 months", "higher response 48% vs 22%", "side effects 18%"], "must_not_invent": ["FDA approval granted"], "constraints": ["exactly 3 bullet points", "strictly avoid medical acronyms PFS OS ORR"], "quantities": ["3 bullet points"]}),
        ("sum_009", EvaluationDifficulty.MEDIUM, "Provide a 1-sentence TL;DR summary of this customer review:\n\nThe noise cancellation on these headphones is exceptional on public transit, and the battery life easily lasted through my entire 14-hour flight. However, the ear cushions became uncomfortably warm after about two hours of continuous use.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["TL;DR", "great noise cancellation and battery", "warm ear cushions discomfort"], "must_not_invent": ["poor microphone audio quality"], "constraints": ["exactly 1 sentence"], "quantities": ["1 sentence"]}),
        ("sum_010", EvaluationDifficulty.EASY, "Summarize this policy into a simple bullet list:\n\nEmployees may expense up to $50 per day for travel meals. Receipts are required for all expenses above $15. Alcohol is non-reimbursable under any circumstances.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["$50 per day meal expense", "receipts above $15", "alcohol non-reimbursable"], "must_not_invent": ["hotel allowance"], "constraints": ["bullet list"], "quantities": ["$50", "$15"]}),
        ("sum_011", EvaluationDifficulty.HARD, "Extract all numeric metrics and dates from the following press release into a structured bullet list:\n\nOn October 12, 2024, GreenVolt announced completion of its 250-megawatt solar farm in Nevada. The facility will generate an estimated 620 gigawatt-hours annually, powering approximately 55,000 homes. Construction took 18 months and came in at a total cost of $185 million, creating 320 temporary jobs.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["October 12, 2024", "250-megawatt", "620 gigawatt-hours", "55,000 homes", "18 months", "$185 million", "320 jobs"], "must_not_invent": ["subsidies from federal grant"], "constraints": ["extract metrics and dates into structured list"], "quantities": ["250-megawatt", "620 gigawatt-hours", "55,000", "18 months", "$185 million", "320"]}),
        ("sum_012", EvaluationDifficulty.MEDIUM, "Summarize this bug report history into a chronological timeline of events:\n\nAt 09:15 UTC on March 4, alerts triggered for elevated 500 error rates on user logins. At 09:30 UTC, on-call engineer identified bad config push to auth microservice. At 09:45 UTC, rollback was initiated. By 10:05 UTC, login error rate returned to normal baseline (0.02%).",
         {"task_family": "SUMMARIZATION", "must_preserve": ["09:15 alert triggered", "09:30 bad config identified", "09:45 rollback initiated", "10:05 resolved"], "must_not_invent": ["data breach occurred"], "constraints": ["chronological timeline format"], "quantities": []}),
        ("sum_013", EvaluationDifficulty.EASY, "Summarize this recipe into a 3-step preparation guide:\n\nFirst, whisk 2 eggs with a pinch of salt and black pepper in a small bowl. Next, melt 1 tablespoon of butter in a non-stick skillet over medium-low heat. Finally, pour in eggs and gently stir with a spatula until soft curds form, taking about 3 minutes.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["whisk eggs with salt and pepper", "melt butter in skillet", "cook soft curds 3 minutes"], "must_not_invent": ["grate cheddar cheese"], "constraints": ["exactly 3 steps"], "quantities": ["3-step"]}),
        ("sum_014", EvaluationDifficulty.HARD, "Condense this legal disclaimer into plain language under 50 words explaining user rights and data ownership:\n\nBy accessing or utilizing the platform, the user acknowledges and agrees that all uploaded media remains the intellectual property of said user, subject to a worldwide, royalty-free license granted to the company for the limited operational purpose of hosting, caching, and serving said content.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["user owns uploaded media", "company gets royalty-free license to host content", "plain language"], "must_not_invent": ["company owns copyright"], "constraints": ["under 50 words"], "quantities": ["50 words"]}),
        ("sum_015", EvaluationDifficulty.MEDIUM, "Synthesize these three customer survey excerpts into 2 overarching customer pain points:\n\nReview 1: 'The checkout form kept declining my credit card without giving any specific error code.'\nReview 2: 'I tried paying with three different cards and each time it just refreshed the page without telling me what went wrong.'\nReview 3: 'Customer support took over 48 hours to answer my billing question.'",
         {"task_family": "SUMMARIZATION", "must_preserve": ["pain point 1: silent payment failure / lack of error messaging", "pain point 2: slow customer support response times"], "must_not_invent": ["shipping delays"], "constraints": ["exactly 2 pain points"], "quantities": ["2 pain points"]}),
        ("sum_016", EvaluationDifficulty.EASY, "Give me a 1-paragraph summary of this paragraph:\n\nPhotosynthesis is the biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. In most cases, oxygen is released as a byproduct.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["plants convert light energy to chemical energy", "carbohydrates from CO2 and water", "oxygen byproduct"], "must_not_invent": ["cellular respiration in mitochondria"], "constraints": ["exactly 1 paragraph"], "quantities": ["1 paragraph"]}),
        ("sum_017", EvaluationDifficulty.MEDIUM, "Summarize the pros and cons of remote work discussed in this text into two balanced lists:\n\nRemote work eliminates daily commuting, providing workers with an average of 54 saved minutes each day and greater personal flexibility. However, many employees report feelings of isolation and blurred boundaries between professional and home life, with 38% noting increased difficulty disconnecting in the evenings.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["pros: no commute, 54 saved minutes, flexibility", "cons: isolation, blurred boundaries, disconnecting difficulty 38%"], "must_not_invent": ["productivity metrics drop by half"], "constraints": ["two balanced lists"], "quantities": []}),
        ("sum_018", EvaluationDifficulty.HARD, "Summarize this 10-page document into key findings [document not attached].",
         {"task_family": "SUMMARIZATION", "must_preserve": ["summary"], "must_not_invent": ["financial disclosure numbers"], "constraints": [], "quantities": [], "source_required": True, "clarification_required": True, "clarification_dimensions": ["missing 10-page document"]}),
        ("sum_019", EvaluationDifficulty.EASY, "Summarize the 3 main arguments from this debate excerpt in one sentence each:\n\nSpeaker A argues that nuclear energy is essential for continuous baseline zero-emission power. Speaker B contends that catastrophic accident risks and unaddressed millennial waste storage make fission unviable. Speaker C proposes that capital investments should focus solely on rapidly deployable solar and wind installations.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["Speaker A: continuous baseline zero-emission", "Speaker B: accident risk and waste storage", "Speaker C: focus on solar and wind"], "must_not_invent": ["coal power subsidies"], "constraints": ["one sentence per argument"], "quantities": ["3 arguments"]}),
        ("sum_020", EvaluationDifficulty.MEDIUM, "Condense this release notice into a user-facing bulletin under 60 words:\n\nEffective Monday at 02:00 AM EST, the portal will undergo planned maintenance for database upgrades lasting 90 minutes. Users will be unable to log in or process transactions during this maintenance window. All automated email notifications will be queued and sent following completion.",
         {"task_family": "SUMMARIZATION", "must_preserve": ["maintenance Monday 2:00 AM EST", "lasting 90 minutes", "no login or transactions", "emails queued"], "must_not_invent": ["password reset required"], "constraints": ["under 60 words"], "quantities": ["60 words", "90 minutes"]}),
    ]
    add_specs("SUMMARIZATION", summarization_specs)

    # Category 11: GENERAL (20 cases)
    general_specs = [
        ("gen_001", EvaluationDifficulty.EASY, "What are the primary differences between renewable and non-renewable energy sources? Give 2 examples of each.",
         {"task_family": "GENERAL", "must_preserve": ["differences between renewable and non-renewable energy", "2 examples of each"], "must_not_invent": ["nuclear fusion reactor costs"], "constraints": [], "quantities": ["2 examples each"]}),
        ("gen_002", EvaluationDifficulty.MEDIUM, "Compare the advantages and disadvantages of electric vehicles (EVs) versus internal combustion engine (ICE) vehicles from economic and environmental perspectives.",
         {"task_family": "GENERAL", "must_preserve": ["electric vehicles vs internal combustion", "economic perspective", "environmental perspective", "advantages and disadvantages"], "must_not_invent": ["hydrogen fuel cell aircraft"], "constraints": [], "quantities": []}),
        ("gen_003", EvaluationDifficulty.HARD, "Provide a balanced, non-prescriptive analysis of the societal tradeoffs of artificial intelligence in healthcare diagnostics, addressing diagnostic accuracy, algorithmic bias, and patient data privacy.",
         {"task_family": "GENERAL", "must_preserve": ["AI in healthcare diagnostics", "diagnostic accuracy", "algorithmic bias", "patient data privacy", "balanced non-prescriptive"], "must_not_invent": ["recommendation to ban AI"], "constraints": ["balanced, non-prescriptive tone"], "quantities": []}),
        ("gen_004", EvaluationDifficulty.EASY, "Give me tips for good sleep hygiene. Under 150 words.",
         {"task_family": "GENERAL", "must_preserve": ["tips for sleep hygiene"], "must_not_invent": ["prescription sleeping pills recommendation"], "constraints": ["under 150 words"], "quantities": ["150 words"]}),
        ("gen_005", EvaluationDifficulty.MEDIUM, "Explain why leaves change color in the autumn, detailing chlorophyll breakdown and carotenoid pigments.",
         {"task_family": "GENERAL", "must_preserve": ["why leaves change color in autumn", "chlorophyll breakdown", "carotenoid pigments"], "must_not_invent": ["tree fungal infection"], "constraints": [], "quantities": []}),
        ("gen_006", EvaluationDifficulty.EASY, "help me with my life",
         {"task_family": "GENERAL", "must_preserve": ["life advice"], "must_not_invent": ["psychiatric diagnosis"], "constraints": [], "quantities": [], "clarification_required": True, "clarification_dimensions": ["career, personal finance, or wellness", "current challenge or goal"]}),
        ("gen_007", EvaluationDifficulty.HARD, "Analyze the economic impact of universal basic income (UBI) pilots across poverty alleviation, labor market participation, and fiscal sustainability. Present arguments from both proponents and critics.",
         {"task_family": "GENERAL", "must_preserve": ["universal basic income UBI", "poverty alleviation", "labor market participation", "fiscal sustainability", "proponents and critics"], "must_not_invent": ["partisan political endorsements"], "constraints": ["balanced presentation of proponents and critics"], "quantities": []}),
        ("gen_008", EvaluationDifficulty.MEDIUM, "What are 4 practical strategies for reducing household plastic waste without making major lifestyle sacrifices?",
         {"task_family": "GENERAL", "must_preserve": ["4 practical strategies", "reducing household plastic waste", "without major lifestyle sacrifices"], "must_not_invent": ["zero-waste homesteading"], "constraints": [], "quantities": ["4 strategies"]}),
        ("gen_009", EvaluationDifficulty.EASY, "How does the water cycle work? Explain evaporation, condensation, precipitation, and collection in 4 clear steps.",
         {"task_family": "GENERAL", "must_preserve": ["water cycle", "evaporation", "condensation", "precipitation", "collection"], "must_not_invent": ["desalination plant operations"], "constraints": ["4 clear steps"], "quantities": ["4 steps"]}),
        ("gen_010", EvaluationDifficulty.HARD, "Examine the philosophical debate between consequentialism and deontology regarding autonomous vehicle collision algorithms (the Trolley Problem in AI). Neutral tone.",
         {"task_family": "GENERAL", "must_preserve": ["consequentialism vs deontology", "autonomous vehicle collision algorithms", "Trolley Problem in AI", "neutral tone"], "must_not_invent": ["definitive moral verdict"], "constraints": ["neutral tone"], "quantities": []}),
        ("gen_011", EvaluationDifficulty.MEDIUM, "Compare urban living versus rural living across cost of living, community connection, access to healthcare, and environmental footprint.",
         {"task_family": "GENERAL", "must_preserve": ["urban vs rural living", "cost of living", "community connection", "healthcare access", "environmental footprint"], "must_not_invent": ["real estate agent contact info"], "constraints": [], "quantities": []}),
        ("gen_012", EvaluationDifficulty.EASY, "What causes ocean tides? Explain the gravitational pull of the moon and the sun in simple terms.",
         {"task_family": "GENERAL", "must_preserve": ["what causes ocean tides", "gravitational pull of moon and sun", "simple terms"], "must_not_invent": ["hydroelectric tidal turbine blueprints"], "constraints": [], "quantities": []}),
        ("gen_013", EvaluationDifficulty.MEDIUM, "Discuss the pros and cons of social media on teenage mental health, citing psychological findings without taking an extreme stance.",
         {"task_family": "GENERAL", "must_preserve": ["social media pros and cons", "teenage mental health", "psychological findings"], "must_not_invent": ["call to ban smartphones"], "constraints": ["without taking an extreme stance"], "quantities": []}),
        ("gen_014", EvaluationDifficulty.HARD, "Analyze the geopolitical and economic consequences of global semiconductor supply chain concentration in East Asia, detailing risks and national diversification policies.",
         {"task_family": "GENERAL", "must_preserve": ["semiconductor supply chain", "concentration in East Asia", "geopolitical and economic risks", "national diversification policies"], "must_not_invent": ["military conflict predictions"], "constraints": [], "quantities": []}),
        ("gen_015", EvaluationDifficulty.EASY, "How do noise-cancelling headphones work? Explain constructive and destructive wave interference.",
         {"task_family": "GENERAL", "must_preserve": ["noise-cancelling headphones", "constructive and destructive wave interference"], "must_not_invent": ["Bluetooth codec bitrates"], "constraints": [], "quantities": []}),
        ("gen_016", EvaluationDifficulty.MEDIUM, "What are the main causes and consequences of biodiversity loss in tropical rainforests? Detail deforestation and habitat fragmentation.",
         {"task_family": "GENERAL", "must_preserve": ["biodiversity loss in tropical rainforests", "deforestation", "habitat fragmentation", "consequences"], "must_not_invent": ["ecotourism vacation packages"], "constraints": [], "quantities": []}),
        ("gen_017", EvaluationDifficulty.MEDIUM, "Explain the origins of the standard 5-day, 40-hour work week, detailing Henry Ford and the labor union movement.",
         {"task_family": "GENERAL", "must_preserve": ["origins of 5-day 40-hour work week", "Henry Ford", "labor union movement"], "must_not_invent": ["4-day work week modern legislation"], "constraints": [], "quantities": ["5-day", "40-hour"]}),
        ("gen_018", EvaluationDifficulty.HARD, "Discuss the ethical considerations surrounding space colonization, including planetary protection, resource allocation vs Earth stewardship, and governance.",
         {"task_family": "GENERAL", "must_preserve": ["ethical considerations of space colonization", "planetary protection", "resource allocation vs Earth stewardship", "governance"], "must_not_invent": ["warp drive physics"], "constraints": [], "quantities": []}),
        ("gen_019", EvaluationDifficulty.EASY, "Why is the sky blue during the day and red during sunset? Explain Rayleigh scattering in 3 simple sentences.",
         {"task_family": "GENERAL", "must_preserve": ["why sky is blue during day and red at sunset", "Rayleigh scattering"], "must_not_invent": ["Ozone layer depletion"], "constraints": ["exactly 3 simple sentences"], "quantities": ["3 sentences"]}),
        ("gen_020", EvaluationDifficulty.MEDIUM, "Compare the nutritional differences and environmental footprints of plant-based milk alternatives (oat, almond, soy, and dairy milk).",
         {"task_family": "GENERAL", "must_preserve": ["plant-based milk alternatives vs dairy", "oat, almond, soy", "nutritional differences", "environmental footprints"], "must_not_invent": ["brand advertisements"], "constraints": [], "quantities": []}),
    ]
    add_specs("GENERAL", general_specs)

    return cases


def build_and_freeze_benchmark(output_dir: str = os.path.join("benchmarks", "v1")) -> Dict[str, str]:
    """Generates the benchmark cases, writes to benchmark.jsonl, computes hash, and saves metadata.json."""
    os.makedirs(output_dir, exist_ok=True)
    cases = generate_benchmark_cases()
    jsonl_path = os.path.join(output_dir, "benchmark.jsonl")

    # Deterministic write
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")

    # Compute SHA-256
    with open(jsonl_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    # Category and difficulty distributions
    cat_dist: Dict[str, int] = {}
    diff_dist: Dict[str, int] = {}
    tags_dist: Dict[str, int] = {}

    for c in cases:
        cat_dist[c.category] = cat_dist.get(c.category, 0) + 1
        diff_dist[c.difficulty.value] = diff_dist.get(c.difficulty.value, 0) + 1
        for t in c.tags:
            tags_dist[t] = tags_dist.get(t, 0) + 1

    metadata = BenchmarkMetadata(
        version="humanintent-benchmark-v1.0",
        frozen_at=datetime.now(timezone.utc).isoformat(),
        sha256=sha256,
        total_cases=len(cases),
        category_distribution=cat_dist,
        difficulty_distribution=diff_dist,
        tags_summary=tags_dist,
    )

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))

    # Write rubric.md
    rubric_path = os.path.join(output_dir, "rubric.md")
    rubric_content = (
        f"# HumanIntent Benchmark v1.0 Evaluation Rubric\n\n"
        f"**Benchmark Version:** humanintent-benchmark-v1.0  \n"
        f"**SHA-256 Digest:** `{sha256}`  \n"
        f"**Total Frozen Cases:** {len(cases)}  \n\n"
        f"## Philosophy\n"
        f"Evaluation in HumanIntent is strictly **invariant-based**. There is no single 'gold' string answer. "
        f"Prompts can be enhanced effectively in multiple ways, provided they adhere to core invariant properties:\n"
        f"1. **Intent & Requirement Preservation**: All items in `must_preserve` must survive.\n"
        f"2. **Constraint & Quantity Preservation**: Negative restrictions and numeric caps must survive.\n"
        f"3. **Zero Hallucination / Anti-Invention**: Elements in `must_not_invent` must NOT be injected.\n"
        f"4. **Correct Clarification Behavior**: Clarification must ONLY be requested when `clarification_required == True`.\n"
        f"5. **Source Dependency Integrity**: If `source_required == True` and source is missing, the system must not pretend the artifact exists.\n"
    )
    with open(rubric_path, "w", encoding="utf-8") as f:
        f.write(rubric_content)

    return {
        "jsonl_path": jsonl_path,
        "metadata_path": meta_path,
        "rubric_path": rubric_path,
        "sha256": sha256,
        "total_cases": str(len(cases)),
    }


if __name__ == "__main__":
    res = build_and_freeze_benchmark()
    print(f"Frozen benchmark generated: {res['total_cases']} cases. SHA-256: {res['sha256']}")
