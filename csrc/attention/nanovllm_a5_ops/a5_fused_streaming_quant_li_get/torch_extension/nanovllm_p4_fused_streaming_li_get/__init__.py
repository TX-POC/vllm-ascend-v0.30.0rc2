"""Isolated P4.1 bridge for one raw-Q/K streaming LI scan.

The backing OPP is deliberately the separately built, source-derived
``A5StreamingQuantLiGet`` closure.  This package adds no vLLM route and never
claims its sparse indices are authoritative until strict equivalence succeeds.
"""

from .bridge import (  # noqa: F401
    AUDIT_CANDIDATE_MARKER,
    AUDIT_QUEUE_LEASE_SENTINEL,
    AUDIT_SCORE_RECEIPT_QUEUE_LEASE_SENTINEL,
    SCORE_RECEIPT_CANDIDATE_BIT,
    SCORE_RECEIPT_COLD_BIT,
    SCORE_RECEIPT_MAPPING_VALID_BIT,
    SCORE_RECEIPT_SCORE_MASK,
    SCORE_RECEIPT_SOURCE_UNMAPPED_BIT,
    SCORE_RECEIPT_TAG,
    SCORE_RECEIPT_TAG_MASK,
    SCORE_RECEIPT_THRESHOLD_PASS_BIT,
    launch,
)
