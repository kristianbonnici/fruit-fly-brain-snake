# Public source release

This is a new release repository assembled from the working viewer and selected research source. Repository name: `fruit-fly-brain-snake`. The original research checkout, local showcase, frozen checkpoints and published website were not modified by this preparation.

## Included

- Current viewer, including the refined layout, cell ranking, nullable selection, and model explanation.
- Two selected full-network replay manifests and an immutable data-download index.
- Portable Python local server and verified on-demand/offline cache.
- Selected model source and its local Python dependencies, with original hashes.
- A visitor README, architecture/model/result/data guides and selected historical reports.
- MIT code licence, preserved third-party notices, citation metadata and contribution guide.
- Portable tests and a GitHub Actions workflow.

## Publication boundary

No old Git history, deployment account files, caches, raw datasets, training corpora, or binary checkpoints were copied into Git. The README links to the public interactive demo. Local downloads require a separately configured compatible host; no default data hostname is embedded in the download code. The local review server can reuse an existing verified cache.

This supports a runnable replay demo and inspectable model implementation. Full historical training reproduction is not supported: the original artifact chain is not bundled and some historical data was pruned. This is stated in the README and model guide.

## Distribution

The source is released under MIT, with separate attribution for MaleCNS data and bundled Three.js. Copyright/citation attribution is “Fruit Fly Brain Snake contributors”. No recording cache is bundled in Git.

An optional future data archive can distribute the immutable chunks while preserving the existing hashes. That is separate from this source release. Local verification is recorded in [verification.md](verification.md); the repository's Checks workflow runs portable tests on GitHub.

## Future work

A portable frozen-model inference bundle, a reproducible new training recipe, and independent performance/causal studies are separate follow-up projects. This release does not silently replace missing training artifacts or generate new experimental results.
