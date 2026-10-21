import json
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(repo_root, "backend")
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.compare_phase10_runs import generate_full_comparison_report

r = generate_full_comparison_report()

print("=" * 90)
print("PHASE 10 VS PHASE 10.1 POST-HARDENING COMPARISON")
print("=" * 90)
header = f"{'Metric':<32} | {'Phase 10':<12} | {'Phase 10.1':<12} | {'Abs Delta':<12} | {'Rel Delta':<10} | {'Verdict':<10}"
print(header)
print("-" * len(header))
for row in r['comparison_table']:
    print(f"{row['metric']:<32} | {row['old']:<12} | {row['new']:<12} | {row['delta']:<12} | {row['relative']:<10} | {row['verdict']:<10}")

print("\n" + "=" * 90)
print("PERFORMANCE & LATENCY ACCOUNTING")
print("=" * 90)
for k, v in r['performance'].items():
    print(f"  {k:<28}: {v}")

print("\n" + "=" * 90)
print("FAILURE TAXONOMY")
print("=" * 90)
print(f"{'Failure Type':<28} | {'Old (Phase 10)':<15} | {'New (Phase 10.1)':<15} | {'Delta':<10}")
print("-" * 75)
for f in r['failure_taxonomy']:
    print(f"{f['failure_type']:<28} | {f['old_count']:<15} | {f['new_count']:<15} | {f['delta']:+d}")

print("\n" + "=" * 90)
print("RELEASE GATES EVALUATION")
print("=" * 90)
print(f"{'Gate Name':<30} | {'Target':<12} | {'Phase 10':<10} | {'Phase 10.1':<10} | {'Status':<10}")
print("-" * 80)
old_gates_map = {g['gate_name']: g['actual_value'] for g in json.load(open('evaluation_results/full_v1/release_gates.json'))}
for g in r['release_gates']:
    og = old_gates_map.get(g['gate_name'], 'N/A')
    og_str = f"{og:.2f}%" if isinstance(og, float) else str(og)
    ng_str = f"{g['actual_value']:.2f}%"
    print(f"{g['gate_name']:<30} | {g['target_threshold']:<12} | {og_str:<10} | {ng_str:<10} | {g['status']:<10}")

print("\n" + "=" * 90)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (1000 Resamples)")
print("=" * 90)
cis = r['new_summary'].get('confidence_intervals', {})
print(f"{'Metric':<32} | {'Point Est':<10} | {'95% CI Lower':<14} | {'95% CI Upper':<14}")
print("-" * 75)
for k, v in cis.items():
    pt = f"{v.get('point_estimate', 0.0):.4f}"
    lo = f"{v.get('ci_lower', 0.0):.4f}"
    hi = f"{v.get('ci_upper', 0.0):.4f}"
    print(f"{k:<32} | {pt:<10} | {lo:<14} | {hi:<14}")

print("\n" + "=" * 90)
print("CONFUSION MATRIX (11x11)")
print("=" * 90)
cm = r['confusion_matrix']
cats = cm.get('labels', [])
print(f"{'Actual \\ Predicted':<18} " + " ".join(f"{c[:4]:>5}" for c in cats))
matrix = cm.get('matrix', [])
for i, c in enumerate(cats):
    row_str = " ".join(f"{matrix[i][j]:>5}" for j in range(len(cats)))
    print(f"{c:<18} {row_str}")

print("\n" + "=" * 90)
print("CATEGORY BREAKDOWN")
print("=" * 90)
print(f"{'Category':<18} | {'Old Acc':<10} | {'New Acc':<10} | {'Old Req':<10} | {'New Req':<10} | {'Old Const':<10} | {'New Const':<10} | {'Old Qty':<10} | {'New Qty':<10}")
print("-" * 95)
for c in r['category_breakdown']:
    print(f"{c['category']:<18} | {c['old_acc']:<10} | {c['new_acc']:<10} | {c['old_req']:<10} | {c['new_req']:<10} | {c['old_const']:<10} | {c['new_const']:<10} | {c['old_qty']:<10} | {c['new_qty']:<10}")

print("\n" + "=" * 90)
print("DIFFICULTY BREAKDOWN")
print("=" * 90)
print(f"{'Difficulty':<12} | {'Old Acc':<10} | {'New Acc':<10} | {'Old Req':<10} | {'New Req':<10} | {'Old Const':<10} | {'New Const':<10} | {'Old Unsup':<10} | {'New Unsup':<10}")
print("-" * 95)
for d in r['difficulty_breakdown']:
    print(f"{d['difficulty']:<12} | {d['old_acc']:<10} | {d['new_acc']:<10} | {d['old_req']:<10} | {d['new_req']:<10} | {d['old_const']:<10} | {d['new_const']:<10} | {d['old_unsupp']:<10} | {d['new_unsupp']:<10}")
