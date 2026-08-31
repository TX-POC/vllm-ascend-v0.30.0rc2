#!/usr/bin/env bash
# Compatibility alias. The full runner below is the only P7 assembly owner.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'P7_ASSEMBLY_FINAL status=2 reason=runner_must_be_executed_with_bash_not_sourced\n' >&2
    return 2
fi

set +u
set +o errexit
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
printf 'P7_ASSEMBLY_REDIRECT mode=fresh_canonical_assembly runner=run_p7_1_speculative_executor.sh\n'
exec bash "${SCRIPT_DIR}/run_p7_1_speculative_executor.sh" "$@"
