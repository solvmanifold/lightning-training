# Data artifacts

Run the CPU-only preparation commands from the repository root:

```bash
python3 scripts/prepare_hellaswag.py
python3 scripts/generate_atlas_smoke.py
python3 scripts/verify_data.py
```

`prepare_hellaswag.py` downloads the pinned 10,042-row HellaSwag validation
split to the ignored `data/raw/` directory, verifies its SHA-256 digest, and
rebuilds the committed 100-case chat-MC smoke artifact.

`generate_atlas_smoke.py` requires no network access and no language model. It
derives both requests and oracle tool calls from deterministic latent cases.
The smoke split is intentionally small; it validates formats and scoring before
we build the full experiment dataset described in
[`docs/BEHAVIORAL_COMPRESSION_EXPERIMENT.md`](../docs/BEHAVIORAL_COMPRESSION_EXPERIMENT.md).

Generated manifests contain counts and file hashes. `verify_data.py` checks the
committed artifacts without downloading anything.

HellaSwag is sourced from the
[`rowanz/hellaswag`](https://github.com/rowanz/hellaswag) repository at the
revision recorded in its manifest. The upstream repository is MIT licensed.
