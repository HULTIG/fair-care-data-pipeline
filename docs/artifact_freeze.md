# Artifact freeze record

Recorded before implementation changes on 2026-09-10 (Europe/Lisbon).

- Baseline revision: `4db74694347708392878a8af4e99cf35eddd6f95`
- Baseline branch: `main` tracking `origin/main`
- Existing uncommitted paths: `results/exp1.csv.bak.20260908_131133`,
  `results/exp1.csv.prefixed.20260908_134818`,
  `results/exp2.csv.prefixed.20260908_134818`,
  `results/exp3.csv.prefixed.20260908_134818`, and `review-20260910/`
- Dataset and dependency manifest checksums are recorded by the execution
  environment alongside each run. The initial dependency manifest was
  `requirements.txt`.
- Historical results and manuscript files are retained. They are not eligible
  as revised-study evidence.

The revised runner must write results under a run-specific directory and must
not reuse a prior summary as a current result.
