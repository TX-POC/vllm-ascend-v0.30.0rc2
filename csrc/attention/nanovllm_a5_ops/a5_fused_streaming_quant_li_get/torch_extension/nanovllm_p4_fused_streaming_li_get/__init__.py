"""Isolated P4.1 bridge for one raw-Q/K streaming LI scan.

The backing OPP is deliberately the separately built, source-derived
``A5StreamingQuantLiGet`` closure.  This package adds no vLLM route and never
claims its sparse indices are authoritative until strict equivalence succeeds.
"""

from .bridge import (  # noqa: F401
    AUDIT_CANDIDATE_MARKER,
    AUDIT_QUEUE_LEASE_SENTINEL,
    launch,
)
