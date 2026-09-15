# Data, downloads, and provenance

## Replay data

The Git repository contains UI code and small manifests. The selected replay and connection buffers are losslessly gzip-compressed into 2,623 immutable chunks (635,823,017 bytes, approximately 606 MiB). The loader requests only the data needed for the current view; full connectivity is loaded on demand.

`viewer/data/pack-index.json` maps logical files to chunks and records their exact compressed sizes and SHA-256 values. Recordings and anatomy also carry original content hashes. Body IDs remain strings in the browser. No quantization, interpolation, or new neural execution is used in packaging.

No personal or deployment-specific data hostname is embedded in the repository. Set the `FRUIT_FLY_DATA_URL` environment variable to a host serving the immutable `/data/packed/` files. This source release supplies no default host; use an existing verified cache or a compatible host you have access to. Download availability depends on that host, while the hashes remain fixed. The local server exposes only the viewer and allowlisted chunks, and binds to localhost.

```sh
# Configure FRUIT_FLY_DATA_URL with your compatible data host.
python3 scripts/fetch_data.py
python3 scripts/serve.py --offline
```

Both commands accept `--cache /path/to/cache` to reuse an existing content-addressed chunk directory without copying it. Corrupt or missing files are rejected in offline mode and downloaded/verified again online. To reclaim the download space later, remove `.cache/replay/`; it contains only downloadable data.

If the viewer reports a download failure, check connectivity and retry. A hash mismatch is a failed download, not permission to bypass validation. If the public host is unavailable, an already verified offline cache remains usable.

## Anatomical source

MaleCNS v1.0 is produced by FlyEM / HHMI Janelia, the University of Cambridge, the MRC Laboratory of Molecular Biology, and Google Research. The source dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- [Official project and attribution](https://male-cns.janelia.org/)
- [Dataset downloads](https://male-cns.janelia.org/download/)
- [Official project overview](https://www.janelia.org/project-team/flyem/male-cns-connectome)

This project selects traced neurons, converts the tables, assigns simplified physiological parameters, learns connection efficacies, and renders simulated activity. These modifications are ours and are not endorsed by the dataset creators. Anatomical data and derived buffers retain their data attribution; the repository's MIT licence does not replace it.

## Research artifacts

The replay manifest identifies the original contracts, checkpoint hashes, source recordings, and selection evidence. Those paths are historical provenance references, not a promise that each referenced file is distributed here.

Selected source files were copied unchanged, with hashes in `docs/source-inventory.json`. Public release utilities and documentation are new. `docs/data-summary.json` records the released data inventory. Existing checkpoints, private development folders, deployment IDs, and import credentials are not included as repository setup requirements.
