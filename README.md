# hf003-mps-verify — the MPS SDPA check, run on real Apple Silicon

This repo exists to finish one verification step: running
`verify_mps_sdpa.py` on Apple Silicon.

The script is the one-command on-hardware check from the HF-003 gift for
[huggingface/transformers#44247](https://github.com/huggingface/transformers/issues/44247)
— the MPS SDPA silent-correctness bug (a PyTorch MPS kernel in torch
[2.8, 2.11) silently returns wrong attention output inside a specific
window). The gift — the workaround patch plus this script — lives in
[this gist](https://gist.github.com/1OF1Research/836e0219e0a02ea737739770cf6ae159);
the issue comment that offered it is
[here](https://github.com/huggingface/transformers/issues/44247#issuecomment-6888083259).

The gift's authors have no Apple Silicon. GitHub Actions `macos-14`
runners are Apple Silicon (M1), so this repo runs the check there, in
the open:

- `.github/workflows/verify.yml` runs on `macos-14` (workflow_dispatch
  or push), installs the pinned stack `torch==2.8.0` +
  `transformers==5.23.0` on Python 3.12, first probes the hardware and
  `torch.backends.mps.is_available()`, then runs the script and saves
  its full output as a workflow artifact.
- `verify_mps_sdpa.py` is byte-identical to the file in the gist
  (md5 `aef2bd0dfbae452d7c37a2683e26f9fc`).
- `probe_mps.py` prints the runner's platform and MPS availability —
  the proof the run happened on a live MPS device.

What a healthy result looks like on an affected torch: Part A plain =
**BUG REPRODUCED**, masked = **MATCHES reference**; Part B runs the
transformers dispatch end-to-end on MPS.

`RESULTS.md` records the measured outcome of the run.
