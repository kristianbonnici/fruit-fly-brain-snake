# Architecture

The viewer and scientific implementation are independent.

| Component | Responsibility |
| --- | --- |
| `viewer/app.js` | Controls, recording selection, sensors, inspector and rankings |
| `viewer/brain.js` | Three.js scene, camera, neuron picking and connection overlays |
| `viewer/replay.mjs` | Shared replay clock, recorded-buffer validation and activity samples |
| `viewer/connections.mjs` | Lazy measured adjacency and efficacy inspection |
| `viewer/transport.mjs` | Lossless decompression and integrity checks |
| `viewer/cell-rankings.mjs` | Mean magnitude, variability and top-cell ranking |
| `scripts/serve.py` | Local static server and immutable data cache |
| `model/snake_whole/` | Selected neural, perception, readout and training research code |

A local browser request for a packed chunk is served from a verified disk cache. On a cache miss, the Python server downloads that exact allowlisted object from the host configured through `FRUIT_FLY_DATA_URL`. Without a configured host, cached data remains usable and missing data produces an explicit setup error. It verifies compressed size and hash before making the object available. The browser verifies again, decompresses it and validates the recording's original content hash.

The UI has no MLX dependency. Research GPU kernels use MLX/Metal; a NumPy/SciPy CPU implementation supports numerical checks. Training is never triggered by viewing a replay.

Version suffixes in the scientific source are intentional: the original provenance contracts hash filenames and implementations. The public tree reorganizes documentation and separates the viewer while preserving those source identities. Removing these suffixes would require a separately verified model migration.
