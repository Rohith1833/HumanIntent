# HumanIntent Benchmark v1.0 Evaluation Rubric

**Benchmark Version:** humanintent-benchmark-v1.0  
**SHA-256 Digest:** `17ddbd8f36ad3754d073ceb7bff89005b13f230f14805ed1342bed78f8feae66`  
**Total Frozen Cases:** 220  

## Philosophy
Evaluation in HumanIntent is strictly **invariant-based**. There is no single 'gold' string answer. Prompts can be enhanced effectively in multiple ways, provided they adhere to core invariant properties:
1. **Intent & Requirement Preservation**: All items in `must_preserve` must survive.
2. **Constraint & Quantity Preservation**: Negative restrictions and numeric caps must survive.
3. **Zero Hallucination / Anti-Invention**: Elements in `must_not_invent` must NOT be injected.
4. **Correct Clarification Behavior**: Clarification must ONLY be requested when `clarification_required == True`.
5. **Source Dependency Integrity**: If `source_required == True` and source is missing, the system must not pretend the artifact exists.
