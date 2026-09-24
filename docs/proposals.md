# Proposals

* Proposals to reformulate the critical questions: they are in `docs/cq_proposals.md`, seven proposals on eight questions and six schemes, to be applied only after the v0 baseline.
* Version the gold instead of overwriting it, the current v1.0 and v2.0 after Dumitru's annotation, because the numbers of the diagnosis (0.374 pipeline, 0.565 zero-shot, 0.834 scheme) are anchored to v1.0. Not decided yet.
* Freeze the gold scheme before the runs of phase 2 and collect the scheme corrections found during annotation in a separate list instead of editing them inside items.csv, because the scheme decides which questions are asked while the verdict does not. Not decided yet.
* Let the pilot job skip, when it is resubmitted with the same run id, the entries already complete: those whose planned calls all have an answered line in `raw.jsonl`, as `calls_per_model` of the manifest counts them. Today the job relaunches the server and the smoke test of every entry, and `execute` finds nothing to do for the complete ones: a few minutes of loading per entry on a four-hour job. Worth doing if resubmissions become frequent. Not decided yet.
