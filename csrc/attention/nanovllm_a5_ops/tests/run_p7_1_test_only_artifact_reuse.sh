#!/usr/bin/env bash
# Compatibility name only.  Reusing an arbitrary P7 OPP was the source of
# repeated mixed-revision failures, so every invocation now uses one fresh
# canonical assembly.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'P7_REUSE_FINAL status=2 reason=runner_must_be_executed_with_bash_not_sourced\n' >&2
    return 2
fi

set +u
set +o errexit
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
printf 'P7_REUSE_REDIRECT mode=fresh_canonical_assembly\n'
exec bash "${SCRIPT_DIR}/run_p7_1_speculative_executor.sh" "$@"
