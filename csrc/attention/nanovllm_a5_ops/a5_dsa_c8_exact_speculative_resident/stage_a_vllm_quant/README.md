# P10 Stage-A: product-derived exact-score receipt

This directory is the source-level foundation for the product streaming C8
path.  It is intentionally neither an overlay onto a pre-existing binary nor
an implementation of transport.  Its job is limited to one formal seam in a
fresh sibling of the resolved `VllmQuantLightningIndexer` source:

```text
formal graph-visible exact uint16 score surface
  -> independent score-DMA completion edge
  -> GM raw score-receipt descriptor publish
```

The original exact Top-K data flow remains owned by the derived C8
implementation.  Stage-A never executes score thresholding, `S -> O` lookup,
resident-slot admission, transport, completion polling, owns a permanent AIV
lane, or writes the original sparse result.  A separate forward-owned device
executor claims a published descriptor, reads the exact score window, then
performs `ExactScoreScan -> CandidateCompact -> PostOnly -> CQ drain`.

## Why the source is staged, not patched in place

`tools/stage_product_c8_sibling.sh` accepts an explicit exact operator source
root.  It first runs the repository provenance probe, requires the supported
VllmQuant family, and creates a fresh source-only package.  It does not edit
the input source and deliberately stops before a build: the seam must be
implemented as a reviewed sibling operator with a formal OpDef, tiling budget,
and graph ABI.

This is not a missing implementation shortcut.  It prevents an unsafe textual
injection from becoming a new product binary before the following invariants
are represented in source and tiling:

1. the score surface and receipt ring are separately declared GM resources,
   never pointers into score UB or CANN's opaque workspace;
2. the native score copy completes through an explicit, independent visibility
   edge before a receipt is published;
3. descriptor payload is globally visible before `PUBLISHED` is observable;
4. the only ring-state reuse edge is `DONE -> EMPTY`; and
5. Stage-B's bounded candidate scratch and forward-owned destination claims
   cannot displace a later exact winner.

The two headers under `op_kernel/arch35` are source contracts for that sibling:
they implement the descriptor-only receipt state transitions and deliberately
do not attach themselves to the current C8 class.  The later derivation patch
must publish one descriptor per logical row segment: `(row, source_begin,
valid_count, score_surface_offset_bytes, score_element_bytes, sequence)`.  It
must keep the original score reader and Top-K data flow intact.  Stage-B—not
this seam—then derives candidates from the referenced score window.
