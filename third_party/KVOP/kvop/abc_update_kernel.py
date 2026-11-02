# SPDX-License-Identifier: Apache-2.0
"""One launch for all cache groups and active requests; no KV payload movement."""

import triton
import triton.language as tl


@triton.jit
def update_abc_kernel(
    T2S,
    S2T,
    DESC,
    ROWS: tl.constexpr,
    POOL: tl.constexpr,
    TOKENS: tl.constexpr,
    HISTORY: tl.constexpr,
    TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    row = tl.program_id(1)
    desc = (group * ROWS + row) * 4
    layer = tl.load(DESC + desc).to(tl.int64)
    owner = tl.load(DESC + desc + 1).to(tl.int64)
    begin = tl.load(DESC + desc + 2)
    end = tl.load(DESC + desc + 3)
    base = (layer * POOL + owner) * TOKENS
    lane = tl.arange(0, BLOCK)
    token = begin + lane
    tl.store(T2S + base + token, HISTORY + token % TAIL, (lane < TAIL) & (token < end))
    # Compute reverse values by ring position rather than reset + scatter,
    # avoiding overlapping stores within the kernel.
    ring_token = begin + (lane - begin % TAIL + TAIL) % TAIL
    tl.store(S2T + base + HISTORY + lane, tl.where(ring_token < end, ring_token, -1), lane < TAIL)
