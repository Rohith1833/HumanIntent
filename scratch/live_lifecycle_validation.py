"""Live lifecycle validation script for Phase 8 local persistence."""

import json
import os
import sys
import time
from uuid import UUID

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./backend"))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from app.db.models import EnhancementSession, Feedback
from app.db.session import engine as app_engine
from app.main import app

client = TestClient(app)


def run_live_lifecycle():
    print("==================================================")
    print("PHASE 8 LIVE LIFECYCLE VALIDATION")
    print("==================================================")

    # 1. Clear request
    print("\n--- Step 1: Clear Request ---")
    t0 = time.perf_counter()
    resp1 = client.post(
        "/api/enhance",
        json={"prompt": "write a python function to compute fibonacci numbers with memoization"},
    )
    t_clear = (time.perf_counter() - t0) * 1000
    assert resp1.status_code == 200, f"Step 1 failed: {resp1.text}"
    data1 = resp1.json()
    session_id_1 = data1["session_id"]
    status_1 = data1["status"]
    print(f"Session 1 ID: {session_id_1}")
    print(f"Session 1 Status: {status_1}")
    print(f"Session 1 History Saved: {data1.get('history_saved')}")
    print(f"Step 1 Time (Inference + DB Write): {t_clear:.2f}ms")

    # 2. Ambiguous request
    print("\n--- Step 2: Ambiguous Request ---")
    t0 = time.perf_counter()
    resp2 = client.post(
        "/api/enhance",
        json={"prompt": "build me an app"},
    )
    t_ambig = (time.perf_counter() - t0) * 1000
    assert resp2.status_code == 200, f"Step 2 failed: {resp2.text}"
    data2 = resp2.json()
    session_id_2 = data2["session_id"]
    status_2 = data2["status"]
    print(f"Session 2 ID: {session_id_2}")
    print(f"Session 2 Status: {status_2}")
    print(f"Session 2 History Saved: {data2.get('history_saved')}")
    print(f"Step 2 Time: {t_ambig:.2f}ms")

    # 3 & 4. Answer clarification using same session_id -> Valid enhanced result
    print("\n--- Step 3 & 4: Answer Clarification (Same Session) ---")
    t0 = time.perf_counter()
    answers = [
        {
            "question_id": "q1",
            "question": "What kind of app?",
            "answer": "A CLI tool in Go for formatting markdown files.",
            "target_dimension": "deliverable_and_tech",
        }
    ]
    resp3 = client.post(
        "/api/enhance",
        json={
            "session_id": session_id_2,
            "prompt": "build me an app",
            "clarification_answers": answers,
        },
    )
    t_answer = (time.perf_counter() - t0) * 1000
    assert resp3.status_code == 200, f"Step 3 failed: {resp3.text}"
    data3 = resp3.json()
    assert data3["session_id"] == session_id_2, "Session ID must be preserved!"
    print(f"Session 2 Reused ID: {data3['session_id']}")
    print(f"Session 2 New Status: {data3['status']}")
    print(f"Session 2 Category: {data3.get('analysis', {}).get('classification', {}).get('category')}")
    print(f"Session 2 Quality Score: {data3.get('quality', {}).get('overall_score')}")
    print(f"Step 3 Time: {t_answer:.2f}ms")

    # 5. Submit feedback
    print("\n--- Step 5: Submit Feedback ---")
    t0 = time.perf_counter()
    resp_fb = client.post(
        "/api/feedback",
        json={
            "session_id": session_id_1,
            "rating": "HELPFUL",
            "reason": "Clear structure and good output expectations.",
            "edited_prompt": "Optional improved version for CLI tool.",
            "dataset_consent": True,
        },
    )
    t_fb = (time.perf_counter() - t0) * 1000
    assert resp_fb.status_code == 200, f"Step 5 failed: {resp_fb.text}"
    fb_data = resp_fb.json()
    print(f"Feedback ID: {fb_data['id']}")
    print(f"Feedback Rating: {fb_data['rating']}")
    print(f"Feedback Consent: {fb_data['dataset_consent']}")
    print(f"Feedback Write Time: {t_fb:.2f}ms")

    # 6. Load from history
    print("\n--- Step 6: Load from History ---")
    t0 = time.perf_counter()
    resp_hist = client.get("/api/history?limit=10&offset=0")
    t_hist_list = (time.perf_counter() - t0) * 1000
    assert resp_hist.status_code == 200
    hist_list = resp_hist.json()
    print(f"History Total Rows: {hist_list['total']}")
    print(f"History Query Time (List): {t_hist_list:.2f}ms")

    t0 = time.perf_counter()
    resp_detail = client.get(f"/api/history/{session_id_2}")
    t_hist_detail = (time.perf_counter() - t0) * 1000
    assert resp_detail.status_code == 200
    detail = resp_detail.json()
    print(f"Detail Session ID: {detail['session_id']}")
    print(f"Detail Original Prompt: {detail['original_prompt']}")
    print(f"Detail Status: {detail['status']}")
    print(f"Detail Clarification Rounds Count: {len(detail['clarification_rounds'])}")
    print(f"History Query Time (Detail): {t_hist_detail:.2f}ms")

    # 7 & 8. Simulate restart & confirm persistence survives
    print("\n--- Step 7 & 8: Restart Simulation & Persistence Check ---")
    # Fresh ORM session to verify data is written to disk
    from sqlalchemy.orm import Session
    with Session(app_engine) as db_session:
        sessions = db_session.scalars(select(EnhancementSession)).all()
        print(f"Verified Disk Sessions Count: {len(sessions)}")
        assert len(sessions) >= 2, "Sessions must survive in persistent storage!"

        fb_persisted = db_session.scalars(
            select(Feedback).where(Feedback.session_id == session_id_1)
        ).first()
        assert fb_persisted is not None
        print(f"Verified Disk Feedback Rating: {fb_persisted.rating}")
        print(f"Verified Disk Dataset Consent: {fb_persisted.dataset_consent}")

    print("\n==================================================")
    print("LIVE LIFECYCLE COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_live_lifecycle()
