#!/usr/bin/env python
"""
verify_mps_sdpa.py — one-command check for the MPS SDPA silent-correctness bug
(transformers#44247, pytorch/pytorch#174861) and for the workaround in
mps_sdpa_workaround.patch.

    python verify_mps_sdpa.py

What it does
------------
PART A (raw PyTorch, no transformers needed) builds the smallest attention
call inside the bug's window — non-causal, bf16/fp16, query length 4, key
length 2048, head dim 64, no mask — and compares, on MPS:
  1. the plain call (no mask) against a float32 CPU reference;
  2. the same call with the patch's float mask applied, against the reference.
On an affected torch (2.8 <= v < 2.11) the plain call is expected to MISMATCH
and the masked call to MATCH. On torch >= 2.11 both should match (upstream
fix). A masked-call mismatch on any version means the workaround shape does
not fix the bug on this hardware — please report that on the issue.

PART B (only if transformers is importable) reports whether the installed
transformers carries the workaround, and runs `sdpa_attention_forward`
end-to-end against a manual reference on whatever device is available.

Exit codes: 0 = ran (verdicts printed), 3 = no MPS device on this machine
(Part A hardware verdicts skipped; the CPU-side checks still run).
"""
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)

Q_LEN, KV_LEN, HEADS, HEAD_DIM = 4, 2048, 4, 64
TOL = 5e-2  # bf16/fp16 kernel-vs-fp32-reference slack; the bug's errors are far larger


def reference(q, k, v):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)


def part_a():
    print("PART A — raw SDPA in the bug window")
    print(f"  torch {torch.__version__}")
    if not torch.backends.mps.is_available():
        print("  SKIP: no MPS device on this machine — Part A needs Apple Silicon.")
        return False
    for dtype in (torch.bfloat16, torch.float16):
        q = torch.randn(1, HEADS, Q_LEN, HEAD_DIM)
        k = torch.randn(1, HEADS, KV_LEN, HEAD_DIM)
        v = torch.randn(1, HEADS, KV_LEN, HEAD_DIM)
        ref = reference(q, k, v)
        qm, km, vm = (t.to("mps", dtype) for t in (q, k, v))
        plain = F.scaled_dot_product_attention(qm, km, vm, is_causal=False).float().cpu()
        err_plain = (plain - ref).abs().max().item()
        zeros = torch.zeros((1, 1, Q_LEN, KV_LEN), dtype=dtype, device="mps")
        masked = F.scaled_dot_product_attention(qm, km, vm, attn_mask=zeros, is_causal=False).float().cpu()
        err_masked = (masked - ref).abs().max().item()
        bug = err_plain > TOL
        fixed = err_masked <= TOL
        print(f"  {str(dtype):15s} plain max err {err_plain:.4f} -> {'BUG REPRODUCED' if bug else 'no bug seen'}; "
              f"with float mask {err_masked:.4f} -> {'MATCHES reference' if fixed else 'STILL WRONG'}")
        if bug and not fixed:
            print("  !! The mask workaround does NOT fix the bug on this hardware — please report this on")
            print("     https://github.com/huggingface/transformers/issues/44247")
    return True


def part_b():
    print("PART B — transformers dispatch")
    try:
        import transformers
        from transformers.integrations import sdpa_attention
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP: transformers not importable ({e})")
        return
    print(f"  transformers {transformers.__version__}")
    has_workaround = hasattr(sdpa_attention, "_mps_sdpa_needs_general_path")
    print(f"  workaround present in installed transformers: {has_workaround}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    class FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_key_value_groups = 1
            self.is_causal = False

    dtype = torch.bfloat16
    q = torch.randn(1, HEADS, Q_LEN, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(1, HEADS, KV_LEN, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(1, HEADS, KV_LEN, HEAD_DIM, dtype=dtype, device=device)
    out, _ = sdpa_attention.sdpa_attention_forward(FakeModule(), q, k, v, None, dropout=0.0, is_causal=False)
    ref = reference(q.float().cpu(), k.float().cpu(), v.float().cpu())
    # sdpa_attention_forward returns (batch, q_len, heads, head_dim); transpose back to compare
    got = out.float().cpu().reshape(1, Q_LEN, HEADS, HEAD_DIM).transpose(1, 2)
    err = (got - ref).abs().max().item()
    print(f"  sdpa_attention_forward on {device}: max err vs fp32 reference {err:.4f} "
          f"-> {'MATCHES' if err <= TOL else 'MISMATCH (expected on MPS + affected torch without the patch)'}")


if __name__ == "__main__":
    ran_a = part_a()
    part_b()
    print()
    print("Done. On an affected torch the expected healthy result is: Part A plain = BUG REPRODUCED,")
    print("masked = MATCHES reference; Part B MATCHES once the patch is applied.")
    sys.exit(0 if ran_a else 3)
