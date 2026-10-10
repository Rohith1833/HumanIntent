import os
import sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./backend"))

from sqlalchemy import select
from sqlalchemy.orm import Session
from app.db.models import EnhancementSession, Feedback
from app.db.session import engine

with Session(engine) as db:
    sessions = db.scalars(select(EnhancementSession)).all()
    print(f"Total Disk Sessions: {len(sessions)}")
    for s in sessions[-3:]:
        print(f"  Session ID: {s.id}")
        print(f"  Created: {s.created_at}")
        print(f"  Status: {s.status}")
        print(f"  Original Prompt Preview: {s.original_prompt[:40]}...")
        print(f"  Final Category: {s.final_category}")
        print(f"  Quality Score: {s.final_quality_score}")
        print(f"  Rounds Count: {len(s.clarification_rounds)}")
        print(f"  Attempts Count: {len(s.enhancement_attempts)}")
        if s.feedback:
            print(f"  Feedback Rating: {s.feedback.rating}, Consent: {s.feedback.dataset_consent}")
        print("  ---")
