# Data artifacts

Run the CPU-only preparation commands from the repository root:

```bash
python3 scripts/prepare_hellaswag.py
python3 scripts/generate_atlas_smoke.py
python3 scripts/generate_beacon_json.py
python3 scripts/prepare_beacon_sft.py
python3 scripts/verify_data.py
```

`prepare_hellaswag.py` downloads the pinned 10,042-row HellaSwag validation
split to the ignored `data/raw/` directory, verifies its SHA-256 digest, and
rebuilds the committed 100-case chat-MC smoke artifact. It also creates the
ignored, reproducible full evaluation artifact under `data/derived/`.
The upstream `ind` field is not globally unique; later occurrences receive a
deterministic `-dupN` case-ID suffix while retaining the original index in
metadata.

`generate_atlas_smoke.py` requires no network access and no language model. It
derives both requests and oracle tool calls from deterministic latent cases.
The smoke split is intentionally small; it validates formats and scoring before
we build the full experiment dataset described in
[`docs/BEHAVIORAL_COMPRESSION_EXPERIMENT.md`](../docs/BEHAVIORAL_COMPRESSION_EXPERIMENT.md).

`generate_beacon_json.py` builds the separate canonical-JSON experiment from
latent job specifications. It commits 2,048 training, 256 development, and 512
locked-test cases with disjoint surface-template families. Labels and defaults
are derived mechanically; no language model participates in generation.

`prepare_beacon_sft.py` converts only the frozen training and development cases
to NeMo AutoModel's OpenAI-chat JSONL format under the ignored
`data/derived/beacon_sft/` directory. It uses the compact evaluation prompt,
creates a fixed 16-case overfit artifact, records hashes, and deliberately
includes zero locked-test cases.

Generated manifests contain counts and file hashes. `verify_data.py` checks the
committed artifacts without downloading anything.

HellaSwag is sourced from the
[`rowanz/hellaswag`](https://github.com/rowanz/hellaswag) repository at the
revision recorded in its manifest. The upstream repository is MIT licensed.
