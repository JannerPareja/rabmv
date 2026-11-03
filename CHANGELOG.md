# Changelog

## 0.1.0 (3 Nov 2026)

First public release.

- **Estimator.** `RabMVClassifier`: an explainable boosting machine with an optimal piecewise Newton step by
  dynamic programming, vectorized bags and exact additive explanations. With `missing='marginalize'` it
  marginalizes what a form did not measure, with purified pairwise interactions. `categorical_features` handles
  nominal concepts. Without these options it is bit-for-bit identical to EBMP v1.
- **Registry schema.** TOML catalog and mappings per form and version (`Schema`, `Catalog`, `Mapping`),
  per-case availability (`harmonize`) and the eligibility rule with a common resolution (`evaluate_eligibility`).
- **Audit.** Checks A, B, C, E (`run_checks`) and the optional refinement of check B (`drop_explained_pairs`),
  validated after simulation v2.
- **Theory.** Proofs that purified marginalization is exact for pairs (`docs/theory/purification.md`), with
  numerical checks.
- **Validation.** Simulation v1 (frozen 26–27 Oct 2026) and simulation v2 (frozen 1 Nov 2026), with a
  post-hoc diagnosis of criterion C4.
- **Example.** A public synthetic registry with its schema, and a quickstart.
- **Community.** Contributing guide, code of conduct, issue and pull-request templates; documentation site
  (MkDocs Material) published to GitHub Pages.
- **Packaging.** BSD-3-Clause license, `CITATION.cff`, Zenodo metadata, continuous integration (Ubuntu and
  Windows, Python 3.12–3.14) and publication to PyPI by trusted publishing.

## Development history (before 0.1.0)

- EBMP v1 (Competencia 1) and phases 1–2e (Oct 2026): package `ebmp`, registry-aware schema, availability,
  eligibility, marginalization, categorical features and audit checks.
- 27 Oct 2026: renamed `ebmp` → `rabmv` and translated to English, without changing any computation.
- 0.1.0.dev10 (27 Oct 2026): purified interactions with marginalization.
- Phases 5 and 6 (28 Oct – 1 Nov 2026): development and pre-specified confirmation with real data
  (aggregates only, `development/`).
