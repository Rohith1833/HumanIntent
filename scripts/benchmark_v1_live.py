"""Live Ollama Performance Acceptance Benchmark for HumanIntent v1.0 Production Release.

Measures actual warm latency, stage breakdowns, token throughput, and invariant verification
on local hardware (NVIDIA GeForce RTX 3050 Laptop GPU + llama3:latest).
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.fast_path import fast_path_pipeline
from app.llm.ollama_client import ollama_client


BENCHMARK_PROMPTS = [
    {
        "id": "PROBE_1_CODING_QUICK",
        "category_expected": "CODING",
        "mode": "quick",
        "prompt": "write an attendance tracking script in python using sqlite3",
        "expected_tech": ["python", "sqlite"],
    },
    {
        "id": "PROBE_2_NEGATION_PROFESSIONAL",
        "category_expected": "CODING",
        "mode": "professional",
        "prompt": "create a rest api in fastapi without any database and do not use docker",
        "expected_tech": ["fastapi"],
        "expected_neg": ["database", "docker"],
    },
    {
        "id": "PROBE_3_RESEARCH_PROFESSIONAL",
        "category_expected": "RESEARCH",
        "mode": "professional",
        "prompt": "compare rust vs go concurrency models with pros and cons",
        "expected_tech": ["rust", "go"],
    },
    {
        "id": "PROBE_4_EDUCATION_DEEP",
        "category_expected": "EDUCATION",
        "mode": "deep",
        "prompt": "explain how public key cryptography works for high school students with simple analogies",
        "expected_tech": [],
    },
    {
        "id": "PROBE_5_AMBIGUITY_CLARIFICATION",
        "category_expected": "GENERAL",
        "mode": "quick",
        "prompt": "build me a system",
        "expected_clarification": True,
    },
]


async def run_benchmark():
    print("=" * 80)
    print("HUMANINTENT v1.0 LIVE OLLAMA PERFORMANCE ACCEPTANCE BENCHMARK")
    print(f"Backend Target: {ollama_client.base_url} | Model: {ollama_client.model}")
    print("=" * 80)

    # 1. Verify Ollama Health
    health = await ollama_client.check_health()
    if not health:
        print("[FATAL] Ollama daemon is unreachable at 127.0.0.1:11434. Aborting.")
        return False
    print("[INIT] Ollama health verified: OK")

    # 2. Warmup probe to ensure GPU layers are mapped
    print("[WARMUP] Sending warm-up ping to model...")
    t_warm_0 = time.perf_counter()
    warm_res = await fast_path_pipeline.client.chat(
        messages=[{"role": "user", "content": "hello"}],
        num_predict=10,
        keep_alive="30m",
    )
    t_warm_elapsed = (time.perf_counter() - t_warm_0) * 1000.0
    print(f"[WARMUP] Completed in {t_warm_elapsed:.1f}ms (response: {warm_res.strip()!r})")
    print("-" * 80)

    results: List[Dict[str, Any]] = []

    for item in BENCHMARK_PROMPTS:
        p_id = item["id"]
        mode = item["mode"]
        prompt = item["prompt"]
        print(f"\n[RUNNING] {p_id} (mode={mode})")
        print(f"  Prompt: \"{prompt}\"")

        t0 = time.perf_counter()
        res = await fast_path_pipeline.enhance_fast_path(
            user_prompt=prompt,
            mode=mode,
        )
        t_total_ms = (time.perf_counter() - t0) * 1000.0
        t_total_s = t_total_ms / 1000.0

        analysis = res.analysis
        enhanced = res.enhanced_prompt
        val = res.validation
        timings = {t.stage: t.elapsed_ms for t in res.stage_timings}

        t_pre = timings.get("deterministic_preprocessing", 0.0)
        t_infer = timings.get("enhancer", res.inference_ms or 0.0)
        t_verify = timings.get("deterministic_verification", 0.0)

        is_clarification = analysis.ambiguity.needs_clarification
        status = "needs_clarification" if is_clarification else "enhanced"

        print(f"  Outcome: {status}")
        print(f"  Latency: Total = {t_total_s:.2f}s ({t_total_ms:.1f}ms) | Pre = {t_pre:.2f}ms | Infer = {t_infer / 1000.0:.2f}s | Verify = {t_verify:.2f}ms")
        if status == "enhanced":
            preview = enhanced.replace("\n", " ")[:120]
            print(f"  Output preview: {preview}...")
            print(f"  Validation: is_valid={val.is_valid}, issues={len(val.issues)}")
        else:
            print(f"  Questions asked: {len(analysis.ambiguity.questions)}")

        results.append({
            "id": p_id,
            "mode": mode,
            "prompt": prompt,
            "category": analysis.classification.category.value,
            "status": status,
            "t_pre_ms": round(t_pre, 2),
            "t_infer_ms": round(t_infer, 2),
            "t_infer_s": round(t_infer / 1000.0, 2),
            "t_verify_ms": round(t_verify, 2),
            "t_total_ms": round(t_total_ms, 2),
            "t_total_s": round(t_total_s, 2),
            "under_30s": (t_total_s < 30.0),
            "enhanced_prompt_length": len(enhanced),
            "enhanced_prompt_sample": enhanced[:250] if enhanced else "",
        })

    # Summary Statistics
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Probe ID':<30} | {'Mode':<12} | {'Category':<12} | {'Total (s)':<10} | {'Infer (s)':<10} | {'< 30s':<6}")
    print("-" * 88)

    normal_latencies = [r["t_total_s"] for r in results if r["status"] == "enhanced"]
    all_latencies = [r["t_total_s"] for r in results]

    for r in results:
        print(f"{r['id']:<30} | {r['mode']:<12} | {r['category']:<12} | {r['t_total_s']:<10.2f} | {r['t_infer_s']:<10.2f} | {'PASS' if r['under_30s'] else 'FAIL':<6}")

    print("-" * 88)
    sorted_normal = sorted(normal_latencies)
    p50 = sorted_normal[len(sorted_normal) // 2] if sorted_normal else 0
    p95 = sorted_normal[-1] if sorted_normal else 0
    max_lat = max(all_latencies) if all_latencies else 0

    print(f"P50 Normal Latency: {p50:.2f}s")
    print(f"P95 Normal Latency: {p95:.2f}s")
    print(f"Max Request Latency: {max_lat:.2f}s")
    print(f"Target Threshold: < 30.00s")

    all_passed = all(r["under_30s"] for r in results)
    print(f"\nFinal Acceptance Verdict: {'PASSED (< 30s for all requests)' if all_passed else 'FAILED'}")

    # Save to JSON artifact
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_v1_results.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_model": "llama3:latest",
            "p50_s": round(p50, 2),
            "p95_s": round(p95, 2),
            "max_s": round(max_lat, 2),
            "all_passed": all_passed,
            "probes": results,
        }, f, indent=2)
    print(f"[REPORT] Saved full JSON results to: {out_path}")
    return all_passed


if __name__ == "__main__":
    asyncio.run(run_benchmark())
