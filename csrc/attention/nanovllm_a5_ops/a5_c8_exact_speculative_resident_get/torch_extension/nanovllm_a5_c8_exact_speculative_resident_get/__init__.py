"""Torch registration for the isolated P9 exact-speculative-resident ABI.

Set ``NANOVLLM_A5_EXACT_SPECULATIVE_RESIDENT_OPAPI_LIB`` to the one
``libcust_opapi.so`` emitted by the fresh, SHA-matched P9 OPP package before
invoking this bridge.  The extension intentionally does not search a generic
custom-OPP path: implicit discovery is how stale ABI packages were selected
in earlier experiments.
"""

from . import _C  # noqa: F401
