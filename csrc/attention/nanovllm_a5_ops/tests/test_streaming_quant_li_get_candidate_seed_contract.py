#!/usr/bin/env python3
"""Source contract for the P6 C8-worker CandidateSeed scheduler.

This is deliberately a small static gate.  It prevents a future refactor from
quietly turning the P6 artifact back into either (a) immediate score-seam
PostOnly, or (b) a fixed Get-core / 31+1 role split.  Runtime strictness lives
in ``test_streaming_quant_li_get_post_payload.py`` on a real NPU.
"""

from __future__ import annotations

from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
HOOK = OPS_ROOT / "a5_streaming_quant_li_get/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h"
VECTOR = OPS_ROOT / "a5_streaming_quant_li_get/op_kernel/arch35/quant_lightning_indexer_service_vector.h"
MODE = OPS_ROOT / "a5_streaming_quant_li_get/op_kernel/a5_streaming_quant_li_get_hook_mode.h"
BUILD = OPS_ROOT / "build_nanovllm_ops.sh"


def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def _ordered(text: str, tokens: tuple[str, ...], label: str) -> None:
    position = -1
    for token in tokens:
        next_position = text.find(token, position + 1)
        assert next_position >= 0, f"{label}: missing {token!r}"
        assert next_position > position, f"{label}: order violation at {token!r}"
        position = next_position


def main() -> None:
    hook = HOOK.read_text(encoding="utf-8")
    vector = VECTOR.read_text(encoding="utf-8")
    mode = MODE.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")

    for forbidden in (
        "kGetLaneCount",
        "kProducerCount",
        "roleTrace",
        "group0",
        "31+1",
        "physical AIV",
    ):
        assert forbidden not in hook, f"P6 hook must not contain {forbidden!r}"

    for required in (
        "A5_STREAMING_LI_GET_POST_SCHEDULER_MODE",
        "A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE",
        "NANOVLLM_A5_STREAMING_LI_GET_POST_SCHEDULER",
        "candidate_seed",
        "no fixed Get core=1",
    ):
        assert required in mode or required in build, required

    scan = _between(hook, "void ScanAndCompactTile(", "#endif  // A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE")
    for forbidden in (
        "TryAcquireKvafStreamingBatchGetQueueFromAicore",
        "ClaimStreamingPrivateArenaFromAiv",
        "PublishKvafStreamingBatchGetDescriptorScratch",
        "BeginKvafStreamingBatchGet",
        "privateArenaHead_",
    ):
        assert forbidden not in scan, (
            "CandidateSeed score seam must not perform queue/private-slot/Post work: "
            + forbidden
        )
    for required in (
        "TryAppendCandidateSeed",
        "kCandidateSlotReady",
    ):
        assert required in scan, required
    append = _between(hook, "bool TryAppendCandidateSeed(", "// Revalidate a previously")
    for required in ("ResolveCandidateIdentity", "kCandidateWordObject"):
        assert required in append, required

    # P6 may not grow the near-full C8 Vector-UB closure. Its two 64-record
    # CandidateSeed microbatches exactly reuse P3's legacy 2 KiB source-offset
    # scratch; descriptor capacity remains a separate 256-request transport
    # limit.
    for required in (
        "kCandidateMicrobatchCapacity = 64U",
        "kCandidateMicrobatchCapacity <= kMaxWaveRequests",
        "P6 CandidateSeed ring must exactly reuse P5 source-offset UB scratch",
        "count >= kCandidateMicrobatchCapacity",
        "kCandidateMicrobatchCapacity * kCandidateSeedWords",
        "streamingCandidateSeedsUb_ = streamingSourceOffsetsBuf_.Get<uint32_t>()",
    ):
        assert required in hook or required in vector, required
    assert "streamingCandidateSeedsBuf_" not in vector

    progress = _between(hook, "void ProgressOwnPostAndCq(", "void ObserveCqAfterScoreTile()")
    _ordered(
        progress,
        (
            "RetireAndReleaseIfTerminal(false)",
            "FindReadyCandidateSlot",
            "PostReadyCandidateSlot",
        ),
        "P6 opportunity seam",
    )
    assert "PostReadyCandidateSlot(slot, candidateTileSequence_, false)" in progress
    assert "PostReadyCandidateSlot(slot, static_cast<uint32_t>(cur_s2_idx), false)" not in progress
    post = _between(hook, "bool PostReadyCandidateSlot(", "void FinalizePendingClaims(uint32_t seed_slot)")
    _ordered(
        post,
        (
            "TryAcquireKvafStreamingBatchGetQueueFromAicore",
            "TryStageCandidateSeed",
            "PublishKvafStreamingBatchGetDescriptorScratch",
            "BeginKvafStreamingBatchGet",
            "FinalizePendingClaims(slot)",
        ),
        "P6 PostOnly lifecycle",
    )
    assert "FinishKvafStreamingBatchGet" not in post

    # A terminal CQ is not permission to forget the local wave: the queue
    # becomes reusable only after its owner-word CAS succeeds.  If that CAS
    # fails, P6 must publish READY/FAILED once, retain its nonzero graph-visible
    # lease, latch quarantine, and make no later score tile retry the CAS.
    retire = _between(
        hook,
        "void RetireAndReleaseIfTerminal(bool finish)",
        "void QuarantineTerminalLease()",
    )
    for required in (
        "if (!terminalReadyPublished_)",
        "PublishTerminalReadyState();",
        "terminalReadyPublished_ = true;",
        "if (releaseQuarantined_)",
        "if (!ReleaseTerminalWaveOrQuarantine())",
        "AddTelemetry(queueId, kLeaseReleased, 1);",
        "hasWave_ = false;",
        "terminalReadyPublished_ = false;",
    ):
        assert required in retire, required
    failure = _between(
        retire,
        "if (!ReleaseTerminalWaveOrQuarantine())",
        "AddTelemetry(queueId, kLeaseReleased, 1);",
    )
    assert "hasWave_ = false;" not in failure
    assert "stagedCount_ = 0U;" not in failure
    assert "!enabled_ || releaseQuarantined_" in hook

    # There must be exactly one raw owner-word release site.  It is wrapped
    # in a checked terminal helper; every unposted caller must propagate its
    # bool result rather than silently forgetting an acquired lease.
    raw_release = "ReleaseKvafStreamingBatchGetQueueFromAicore("
    assert hook.count(raw_release) == 1, (
        "every raw queue-release site must route through the checked helper"
    )
    terminal_release = _between(
        hook,
        "bool ReleaseTerminalWaveOrQuarantine()",
        "bool ReleaseUnpostedLease(",
    )
    _ordered(
        terminal_release,
        (
            "if (kvaf_fused_get::ReleaseKvafStreamingBatchGetQueueFromAicore(",
            "return true;",
            "QuarantineTerminalLease();",
            "return false;",
        ),
        "P6 checked terminal queue release",
    )
    quarantine = _between(
        hook,
        "void QuarantineTerminalLease()",
        "bool ReleaseTerminalWaveOrQuarantine()",
    )
    assert "releaseQuarantined_ = true;" in quarantine
    assert "hasWave_ = true;" in quarantine
    assert "hasWave_ = false;" not in quarantine

    unposted_release = _between(
        hook,
        "bool ReleaseUnpostedLease(",
        "void AddTelemetry(",
    )
    assert "return ReleaseTerminalWaveOrQuarantine();" in unposted_release
    unposted_call_lines = [
        line.strip()
        for line in hook.splitlines()
        if "ReleaseUnpostedLease(lease)" in line
    ]
    assert unposted_call_lines
    assert all(
        "if (!ReleaseUnpostedLease(lease))" in line
        for line in unposted_call_lines
    ), "every unposted release caller must check and propagate failure"

    # The score buffer must be released before the later CQ witness.  The
    # source also retains the immediate `Emit` branch for the P3 control
    # binary, so inspect only the P6 conditional branch.
    p6_start = vector.index("streamingKvafHook_.ScanAndCompactTile")
    p6_branch = vector[p6_start:vector.index("DataCopyExtParams copyOutParams", p6_start)]
    _ordered(
        p6_branch,
        (
        "streamingKvafHook_.ScanAndCompactTile",
        "FetchEventID(HardEvent::S_V)",
        "SetFlag<HardEvent::S_V>",
        "WaitFlag<HardEvent::S_V>",
        "streamingKvafHook_.ProgressOwnPostAndCq",
        "streamingKvafHook_.ObserveCqAfterScoreTile",
        ),
        "P6 score-UB release",
    )
    assert "BeginKvafStreamingBatchGet" not in p6_branch

    # The isolated build must retain the canonical macro derivation header;
    # raw mode definitions alone silently preprocess the full P6 closure out.
    for required in (
        'cat "${STREAMING_LI_DIR}/op_kernel/a5_streaming_quant_li_get_hook_mode.h"',
        "streaming LI staged full-post closure derivation mismatch",
        "streaming LI staged CandidateSeed derivation mismatch",
    ):
        assert required in build, required

    print(
        "P6_CANDIDATE_SEED_CONTRACT_PASS "
        "scheduler=worker_local_double_microbatch candidate_microbatch=64 "
        "descriptor_capacity=256 fixed_get_core=0 "
        "score_seam=scan_compact_only post=opportunity_only cq=nonblocking"
    )


if __name__ == "__main__":
    main()
