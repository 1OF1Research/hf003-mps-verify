# Results — the on-hardware run (2026-11-14)

The gift's one open step — run `verify_mps_sdpa.py` on Apple Silicon —
is done. The wall fell: the bug reproduces on the MPS device through
the real transformers dispatch, and the gift's workaround, applied to
the installed transformers, brings the same dispatch back to the
float32 reference.

## Where it ran

GitHub Actions `macos-15` runner, public workflow, re-runnable by
anyone (workflow_dispatch). Runner probe, from the job log:

```
      Chip: Apple M1 (Virtual)
platform: macOS-15.8.2-arm64-arm-64bit
machine: arm64
torch: 2.8.0
mps built: True
mps available: True
```

Pinned stack: Python 3.12.10, `torch==2.8.0` (inside the bug's
[2.8, 2.11) window), `transformers==5.23.0`. The script is
byte-identical to the gist's (md5 `aef2bd0dfbae452d7c37a2683e26f9fc`).

Note on runner choice: `macos-14` never provisioned on this account —
GitHub's annotation reads "The configuration
'macos-14-arm64-us-default' is not supported" — so the runs use
`macos-15`, same Apple Silicon class.

## Run 43671314203 — stock transformers, then with the gift applied

Phase 1, stock `transformers 5.23.0`:

```
PART A — raw SDPA in the bug window
  torch.bfloat16  plain max err 0.5681 -> BUG REPRODUCED; with float mask 0.0009 -> MATCHES reference
  torch.float16   plain max err nan -> no bug seen; with float mask 0.0002 -> MATCHES reference
PART B — transformers dispatch
  workaround present in installed transformers: False
  sdpa_attention_forward on mps: max err vs fp32 reference 0.5814 -> MISMATCH (expected on MPS + affected torch without the patch)
```

Phase 2, after `patch -p2` applied `mps_sdpa_workaround.patch` to the
installed transformers (log: `patching file
'transformers/integrations/sdpa_attention.py'`):

```
PART A — raw SDPA in the bug window
  torch.bfloat16  plain max err 0.8730 -> BUG REPRODUCED; with float mask 0.0009 -> MATCHES reference
  torch.float16   plain max err nan -> no bug seen; with float mask 0.0002 -> MATCHES reference
PART B — transformers dispatch
  workaround present in installed transformers: True
  sdpa_attention_forward on mps: max err vs fp32 reference 0.0005 -> MATCHES
```

Run: https://github.com/1OF1Research/hf003-mps-verify/actions/runs/43671314203
(full job log and the script's own output logs are the run's artifacts)

## Run 43670614450 — first run, stock only

```
PART A — raw SDPA in the bug window
  torch.bfloat16  plain max err 0.8535 -> BUG REPRODUCED; with float mask 0.0009 -> MATCHES reference
  torch.float16   plain max err nan -> no bug seen; with float mask 0.0002 -> MATCHES reference
PART B — transformers dispatch
  workaround present in installed transformers: False
  sdpa_attention_forward on mps: max err vs fp32 reference 0.5736 -> MISMATCH (expected on MPS + affected torch without the patch)
```

Run: https://github.com/1OF1Research/hf003-mps-verify/actions/runs/43670614450

## Reading the numbers honestly

- The bf16 arm carries the clean reproduction: plain error 0.57–0.87
  across runs (the buggy kernel's reduction order varies run to run),
  roughly a thousand times the masked arm's 0.0009. The masked arm —
  the workaround's exact float mask — matched every time, both
  dtypes.
- The fp16 plain call returned NaN (max err `nan`). NaN fails the
  script's `>` threshold test, so the script labels it "no bug seen";
  the number is quoted as the log carries it. A NaN output against a
  finite reference is not a match — but the label is the script's,
  and the bf16 arm is the decisive one.
- Part B is the end-to-end claim: through the real
  `sdpa_attention_forward` on MPS, stock transformers misses at
  0.5814; with the gift's patch applied, the same call matches at
  0.0005 — the healthy bf16 noise floor.
- An earlier check on this runner class with torch 2.9.1 saw the MPS
  device refuse outright (the paravirtual Metal gate) and settled
  nothing. On torch 2.8.0 the device initializes and the kernels
  execute — which is what makes the plain arm's divergence readable
  as the bug rather than as a silent fallback. A physical-Mac run of
  the same one command remains welcome as a second witness.
