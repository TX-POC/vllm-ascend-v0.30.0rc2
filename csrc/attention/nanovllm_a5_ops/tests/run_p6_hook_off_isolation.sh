#!/usr/bin/env bash
# Run the P6 ``hook_off`` isolation gate on an already-built target OPP.
#
# This is intentionally a runner-only helper: it never builds an OPP or a
# bridge.  OPP provenance comes exclusively from each selected artifact's
# manifest, while the Python runner is verified against the fetched branch
# head.  Keeping those two identities separate is essential: a test-only
# commit must not be mistaken for a stale compiled OPP.

set +u
set -o pipefail

_p6_note() {
    printf '%s\n' "$*"
}

_p6_required_path() {
    local name="$1"
    local value="$2"
    if [ -z "$value" ]; then
        _p6_note "P6_HOOK_OFF_BLOCKED reason=missing_${name}"
        return 1
    fi
    return 0
}

VLLM_KIP="${VLLM_KIP:-}"
if ! _p6_required_path vllm_worktree "$VLLM_KIP" || \
   ! git -C "$VLLM_KIP" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=invalid_vllm_worktree path=${VLLM_KIP:-<unset>}"
    exit 2
fi

P6_REFERENCE_OPP_ROOT="${P6_REFERENCE_OPP_ROOT:-}"
P6_TARGET_OPP_ROOT="${P6_TARGET_OPP_ROOT:-}"
P6_KVAF_BUILD="${P6_KVAF_BUILD:-}"
if ! _p6_required_path reference_opp "$P6_REFERENCE_OPP_ROOT" || \
   ! _p6_required_path target_opp "$P6_TARGET_OPP_ROOT" || \
   ! _p6_required_path kvaf_build "$P6_KVAF_BUILD"; then
    exit 2
fi

if [ ! -d "$P6_REFERENCE_OPP_ROOT" ] || [ ! -d "$P6_TARGET_OPP_ROOT" ] || \
   [ ! -d "$P6_KVAF_BUILD" ]; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=artifact_or_kvaf_path_missing"
    _p6_note "P6_HOOK_OFF_PATHS reference=<$P6_REFERENCE_OPP_ROOT> target=<$P6_TARGET_OPP_ROOT> kvaf_build=<$P6_KVAF_BUILD>"
    exit 2
fi

VLLM_DIRTY="$(git -C "$VLLM_KIP" status --porcelain --untracked-files=no 2>/dev/null || true)"
if [ -n "$VLLM_DIRTY" ]; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=tracked_vllm_changes"
    git -C "$VLLM_KIP" status --short
    exit 2
fi

# Fetch first, then compare local HEAD with the actual remote branch ref.
# There is deliberately no caller-supplied expected SHA in this script.
if ! git -C "$VLLM_KIP" fetch origin kip_demo_cs; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=git_fetch_failed"
    exit 2
fi
if ! git -C "$VLLM_KIP" merge --ff-only origin/kip_demo_cs; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=git_fast_forward_failed"
    exit 2
fi

P6_RUNNER_HEAD="$(git -C "$VLLM_KIP" rev-parse HEAD 2>/dev/null || true)"
P6_BRANCH_HEAD="$(git -C "$VLLM_KIP" rev-parse origin/kip_demo_cs 2>/dev/null || true)"
if [ -z "$P6_RUNNER_HEAD" ] || [ "$P6_RUNNER_HEAD" != "$P6_BRANCH_HEAD" ]; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=runner_not_at_fetched_branch runner=${P6_RUNNER_HEAD:-missing} branch=${P6_BRANCH_HEAD:-missing}"
    exit 2
fi
_p6_note "P6_HOOK_OFF_RUNNER head=$P6_RUNNER_HEAD branch_head=$P6_BRANCH_HEAD ready=1"

OPS_ROOT="${OPS_ROOT:-$VLLM_KIP/csrc/attention/nanovllm_a5_ops}"
KVAF_PYTHON="${KVAF_PYTHON:-/usr/local/python3.11.10/bin/python3}"
if [ ! -x "$KVAF_PYTHON" ]; then
    KVAF_PYTHON="$(command -v python3 2>/dev/null || true)"
fi
if [ -z "$KVAF_PYTHON" ] || [ ! -x "$KVAF_PYTHON" ]; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=python_unavailable"
    exit 2
fi

P6_PHYSICAL_NPU_ID="${P6_PHYSICAL_NPU_ID:-7}"
if [ "$P6_PHYSICAL_NPU_ID" != "7" ]; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=physical_npu_must_be_7 got=$P6_PHYSICAL_NPU_ID"
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="$P6_PHYSICAL_NPU_ID"
unset ASCEND_VISIBLE_DEVICES
export P3_PHYSICAL_NPU_ID="$P6_PHYSICAL_NPU_ID"
export P3_NPU_ID=0
export NPU_ID=0
export DEVICE_ID=0

P6_REFERENCE_BRIDGE_ROOT="$OPS_ROOT/a5_official_c8_reference/torch_extension"
P6_TARGET_BRIDGE_ROOT="$OPS_ROOT/a5_streaming_quant_li_get/torch_extension"
export PYTHONPATH="$P6_REFERENCE_BRIDGE_ROOT:$P6_TARGET_BRIDGE_ROOT:$P6_KVAF_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"
export NANOVLLM_A5_REFERENCE_C8_OPP="$P6_REFERENCE_OPP_ROOT"
export NANOVLLM_A5_STREAMING_LI_OPP="$P6_TARGET_OPP_ROOT"
export P3_PUBLICATION_MODE=off
export P3_INLINE_MODE=post
export P3_POST_CLOSURE_MODE=full
export P3_POST_SCHEDULER_MODE=candidate_seed

# This exports the reference/target *artifact* source heads and the current
# runner head separately. It is the only provenance contract used by this
# run; do not add a handwritten expected SHA here.
source "$OPS_ROOT/tests/resolve_streaming_li_get_artifacts.sh" \
    "$P6_REFERENCE_OPP_ROOT" \
    "$P6_TARGET_OPP_ROOT" \
    "$VLLM_KIP"
P6_ARTIFACT_RC=$?
if [ "$P6_ARTIFACT_RC" -ne 0 ]; then
    _p6_note "P6_HOOK_OFF_BLOCKED reason=artifact_environment_resolution_failed rc=$P6_ARTIFACT_RC"
    exit 2
fi

P6_RESULT="${P6_RESULT:-/home/c00982601/results/p6_hook_off_state_npu7_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$P6_RESULT"
P6_HOOK_LOG="$P6_RESULT/hook_off.log"

timeout 240s "$KVAF_PYTHON" \
    "$OPS_ROOT/tests/test_streaming_quant_li_get_post_payload.py" \
    --device npu:0 \
    --physical-npu-id "$P6_PHYSICAL_NPU_ID" \
    --batch-get-kernel "$P6_KVAF_BUILD/bin/kvaf_execute_batch_get_kernels.o" \
    --batch-put-kernel "$P6_KVAF_BUILD/bin/kvaf_execute_batch_put_kernels.o" \
    --post-exercise hook_off \
    >"$P6_HOOK_LOG" 2>&1
P6_HOOK_RC=$?

_p6_note "P6_HOOK_OFF_RESULT status=$P6_HOOK_RC log=$P6_HOOK_LOG"
grep -E \
'^(P3_ARTIFACT_ENV|P3_POST_ISOLATION_STATE|P3_POST_MATH_DIAG|P3_POST_ISOLATION_PASS|P3_POST_PAYLOAD_FAIL|P6_HOOK_OFF_)' \
    "$P6_HOOK_LOG"

exit "$P6_HOOK_RC"
