#!/usr/bin/env bash
# Resolve the immutable source closure of explicitly selected P3 artifacts.
#
# This file is intended to be *sourced* by a scoped NPU validation command:
#
#   source resolve_streaming_li_get_artifacts.sh \
#     "$REFERENCE_OPP_ROOT" "$TARGET_OPP_ROOT" "$VLLM_WORKTREE"
#
# A Python-only test commit can legitimately be newer than a compiled OPP.
# Do not compare the runner worktree head with an OPP manifest.  Instead this
# helper exports the exact source heads recorded by the two selected artifacts
# and records the runner head separately for reproducibility.

_p3_artifact_find_manifest() {
    local root="$1"
    local name="$2"
    local direct="$root/$name"
    local matches=()
    local candidate

    if [ -f "$direct" ]; then
        printf '%s\n' "$direct"
        return 0
    fi
    while IFS= read -r candidate; do
        matches+=("$candidate")
    done < <(find "$root" -type f -name "$name" -print 2>/dev/null | sort)
    if [ "${#matches[@]}" -ne 1 ]; then
        printf 'P3_ARTIFACT_ENV_FAIL reason=manifest_ambiguous root=%s name=%s count=%s\n' \
            "$root" "$name" "${#matches[@]}" >&2
        return 2
    fi
    printf '%s\n' "${matches[0]}"
}

_p3_artifact_value() {
    local manifest="$1"
    local key="$2"
    awk -F= -v expected="$key" '$1 == expected { print substr($0, length(expected) + 2); exit }' \
        "$manifest"
}

_p3_resolve_artifact_environment() {
    if [ "$#" -ne 3 ]; then
        printf 'P3_ARTIFACT_ENV_FAIL reason=usage expected=reference_opp_root,target_opp_root,runner_worktree\n' >&2
        return 2
    fi

    local reference_root="$1"
    local target_root="$2"
    local runner_root="$3"
    local reference_manifest target_manifest
    local reference_head target_head kvaf_head runner_head

    reference_manifest="$(_p3_artifact_find_manifest "$reference_root" P3_REFERENCE_PROVENANCE)" || return $?
    target_manifest="$(_p3_artifact_find_manifest "$target_root" P3_STREAMING_LI_PROVENANCE)" || return $?
    reference_head="$(_p3_artifact_value "$reference_manifest" source_head)"
    target_head="$(_p3_artifact_value "$target_manifest" source_head)"
    kvaf_head="$(_p3_artifact_value "$target_manifest" kvaf_source_head)"
    runner_head="$(git -C "$runner_root" rev-parse HEAD 2>/dev/null || true)"

    if ! printf '%s\n' "$reference_head" | grep -Eq '^[0-9a-f]{40}$' || \
       ! printf '%s\n' "$target_head" | grep -Eq '^[0-9a-f]{40}$' || \
       ! printf '%s\n' "$runner_head" | grep -Eq '^[0-9a-f]{40}$'; then
        printf 'P3_ARTIFACT_ENV_FAIL reason=source_head_invalid reference=%s target=%s runner=%s\n' \
            "${reference_head:-missing}" "${target_head:-missing}" "${runner_head:-missing}" >&2
        return 2
    fi

    export P3_REFERENCE_EXPECTED_OPP_SOURCE_HEAD="$reference_head"
    export P3_TARGET_EXPECTED_OPP_SOURCE_HEAD="$target_head"
    export P3_RUNNER_EXPECTED_HEAD="$runner_head"
    unset P3_REFERENCE_EXPECTED_SOURCE_HEAD P3_TARGET_EXPECTED_SOURCE_HEAD
    if [ "$kvaf_head" = "not_applicable" ] || [ -z "$kvaf_head" ]; then
        unset P3_KVAF_EXPECTED_SOURCE_HEAD
        kvaf_head="not_applicable"
    else
        export P3_KVAF_EXPECTED_SOURCE_HEAD="$kvaf_head"
    fi

    printf 'P3_ARTIFACT_ENV reference_opp_source_head=%s target_opp_source_head=%s runner_head=%s kvaf_source_head=%s\n' \
        "$reference_head" "$target_head" "$runner_head" "$kvaf_head"
}

_p3_resolve_artifact_environment "$@"
_p3_artifact_env_rc=$?
unset -f _p3_artifact_find_manifest _p3_artifact_value _p3_resolve_artifact_environment
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return "$_p3_artifact_env_rc"
fi
exit "$_p3_artifact_env_rc"
