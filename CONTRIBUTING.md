# Contributing

Start with the [architecture](docs/architecture.md) and the [model scope](model/README.md). The viewer runs without a build step: `python3 scripts/serve.py`.

For UI changes, preserve shared-clock synchronization, keyboard access, mobile layouts, and the distinction between recorded values and display transformations. Verify selection, scrubbing, recording switching and missing-coordinate cells. Do not invent intermediate neural samples.

Run `npm test` (Node 22+) and `python3 -m unittest discover -s tests -p 'test_*.py' -v`. Data changes should also pass `npm run test:replays` with the complete cache. Explain what changed and what was tested in each pull request.

Keep large data, checkpoints, caches, local paths, tokens, and deployment account configuration out of Git. New recordings need provenance, hashes, a sampling description, complete terminal-state checks and explicit performance-selection context.

Research changes must distinguish anatomical evidence, modelling assumptions and measured results. Existing versioned source is retained for provenance; introduce and test a deliberate new version when changing the numerical model.

Contributions to original code are accepted under the repository's MIT licence. Preserve third-party licence notices and data attribution.
