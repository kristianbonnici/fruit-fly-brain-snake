# Release verification

Checked locally on 2026-09-16. This is release packaging verification, not a new model training or performance experiment.

- 23 JavaScript tests passed: replay clocks, scrubbing, cancellation, recording/collection switching, identity/buffer validation, fine/coarse sample handling, connection decoding, highlighting, and cell ranking.
- 4 Python download tests passed: offline cache, unknown paths/corruption, atomic download reuse, and invalid response rejection.
- 7 model tests passed; 1 optional Metal test skipped because MLX is not installed in the test environment. Tests include dense-reference and finite-difference numerical comparisons, decoder optimization restoration, and recording invariants.
- Both complete games validated: 157 decisions / 10 food and 220 decisions / 20 food. Original hashes, dimensions, observations, action transitions, growth, and terminal states checked without neural execution. Food respawn positions are checked for legality; the Node validator does not reproduce Python's random-number sequence.
- All 25,563,197 directed rows validated in each of the incoming and outgoing indexes, including learned efficacy overlays and 124,025,046 anatomical synapse counts per index.
- 60 original Python source/test files match their recorded SHA-256 hashes. Versioned source was preserved rather than rewritten.
- Real unauthenticated downloads of the smallest (69-byte) and largest (7,993,020-byte) compressed chunks succeeded with exact size/hash validation. Full replay verification used the existing cache to avoid redundant storage and network traffic.
- The release viewer was visually checked at desktop and 390-pixel mobile widths. Recording switching and the final 20-food state work; mobile document width equals viewport width. Screenshot: [viewer.png](images/viewer.png).
- A clean export containing only staged Git files passed the same portable and model tests. Its local server served the UI and downloaded/verified a missing chunk into a new default cache, with no existing data cache or checkout paths required for the viewer. Model tests reused an installed NumPy/SciPy environment.
- A focused pattern scan found no copied home-directory paths, deployment project identifiers, private-key headers, or common API-token formats. This is a publication hygiene check, not a comprehensive security audit.

The GitHub Actions configuration has not yet executed remotely. Full historical training reproduction and independent biological/causal validation remain outside this release's verified scope.

## Configurable host update

The original hard-coded demo/data URL was subsequently removed. Six downloader tests pass, including environment-based configuration and a missing-host error without a network request. The earlier real-download checks describe the prior configured-host setup. Fresh public setup now awaits the replacement URL; cached/offline viewing remains available.
