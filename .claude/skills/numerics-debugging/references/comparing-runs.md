# Comparing two capture logs

## Real log format

This is the exact output `write_log` produces (verified), one block per op:

```
[features.0.proj/op_0_addmm]
  Shape: (16, 512) torch.float32
  L2: 5.125471e+01
  Min: -2.010448e+00
  Max: 2.068169e+00
  Mean: 3.024476e-03
  Output hash: 3.691091e+03
  Input hashes: 1.675600e+01, 3.217083e+03, 4.089176e+03

[features.0/op_0_relu]
  Shape: (16, 512) torch.float32
  L2: 3.630413e+01
  Min: 0.000000e+00
  Max: 2.068169e+00
  Mean: 2.267985e-01
  Output hash: 1.857934e+03
  Input hashes: 3.691091e+03

[<none>/op_0_sum]
  Shape: () torch.float32
  ...
  Output hash:
```

Notes from real captures:

- The header `module_fqn/op_N_opname` is the match key. `op_N` is a per-module
  counter, so the same op in two runs gets the same key when the module
  structure matches.
- `relu` is attributed to the enclosing block (`features.0`), not the linear, by
  the depth-stack reconstruction — that is correct (it is called in
  the parent module's `forward`, not inside the linear submodule).
- The loss `sum` lands under `<none>` (it runs outside any module) and has an
  **empty** `Output hash` (scalar) — expected, skip it.
- Backward ops carry a trailing `  Phase: backward` line.

## Pairing ops, then diffing

`compare` pairs by exact key and reports the first op whose output hash
differs. The hashes are formatted strings, so **parse them as floats and use a
relative tolerance** rather than `==` — `norm` hashes are float64 L1 and the
last printed digit can wobble on legitimate reduction-order noise:

```python
def _floats(s):
    return [float(x) for x in s.split(",")] if s.strip() else []

def diverges(ha, hb, rtol=1e-6):
    fa, fb = _floats(ha), _floats(hb)
    if len(fa) != len(fb):
        return True
    return any(abs(a - b) > rtol * (abs(a) + abs(b) + 1e-30) for a, b in zip(fa, fb))
```

Pair in passes, most specific first, stopping at the first that matches:

1. **Exact key** — identical `module_fqn/op_N_opname`. Strip the model-class
   root prefix on both sides first (FQNs are rooted at the model class, e.g.
   `Net.features.0...`) so the naming agrees with `named_modules()`.
2. **Fuzzy key** — same FQN and op name but a shifted `op_N` counter (one side
   has an extra in-place collective consuming an early slot). Pair by op-name
   order within the module.
3. **Positional / stats fallback** — when keys don't resolve (e.g. a traced
   run with everything under `<none>`), pair by dispatch order and confirm
   with matching `Shape` + nearest output hash. Treat these pairings as weaker
   evidence and say so.

If eager logs contain many `<none>` keys, recapture with
`annotate_modules=True` before using positional matching. Positional matching
is for cases where module metadata is genuinely unavailable, such as some
compiled or traced executions.

## Reading a divergence

- Compare `Output hash` with the tolerance above. Equal hash + equal `Shape`
  -> treat the op as matching.
- If output hashes differ, check `Input hashes`. **If the inputs already
  differ, the divergence is upstream** — this op only propagates it. Walk back
  to the op that produced those inputs and continue until you find an op whose
  inputs match but whose output differs. That op is the root.
- In-place ops (`_fused_adam_`, names ending in `_`) often have an empty
  `Output hash` (the dispatch returns `None`); compare their `Input hashes`
  instead — a moved input hash is the mutation.
- A single isolated last-digit difference that does not propagate is
  reduction-order noise, not a bug. A difference that **grows** downstream is
  the real divergence — report the earliest op in that chain whose inputs
  still matched.

Verified example (fp32 vs TF32 matmul, same seed/dtype/inputs): all forward
ops matched; the first divergence was a **backward** `mm` whose two input
hashes were identical across runs -> the matmul kernel itself is the source.
Identical-vs-identical runs reported zero divergences across all paired ops
(no false positives).

## Common key mismatch patterns

When comparing eager vs `torch.compile`/`aot_fx_trace`, keys drift for
structural reasons. Express known correspondences as a manual override map
(`{run1_key: run2_key}`) applied before the automatic passes; confirm each by
matching `Shape` and a near-matching output hash first.

- **Recompute FQN drift** — activation checkpointing or another rematerializer
  re-runs modules, so one side logs a shorter FQN (`submodule`) while the other
  keeps the full one (`stage.2.submodule`).
- **Per-module counter shifts** — one side runs an extra bookkeeping,
  communication, or in-place op under a module key, shifting every later
  `op_N` in that module by one.
- **Distributed op lowering / renaming** — an eager collective or device
  transfer can correspond to a traced chain of lower-level ops, often
  re-attributed to the consumer module.
- **Repeated subgraphs** — models with repeated blocks can execute many
  similar `mul`/`add`/`mm`/`bmm` patterns, so fuzzy matching can drift by one
  repeated block.
- **Backward attribution drift** — a backward op can land under `<backward>`,
  a parent module, or a generated backward op name when the grad_fn walk can't
  claim it; pair it explicitly to its eager counterpart.

## Before concluding "real bug"

- Confirm both runs used the **same dtype, seed, inputs**, and determinism
  settings. If not, every row diverges — fix setup and recapture; not a bug.
- Distinguish reduction-order noise (isolated, last-digit, non-propagating)
  from a true divergence (propagates and grows). Report the earliest op whose
  inputs matched but output diverged.
