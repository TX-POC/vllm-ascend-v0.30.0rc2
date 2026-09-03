#!/usr/bin/env bash
# P10 ForwardPrepare is source-only until it is assembled with Stage A and
# Stage B. A separately installed vendor can make the bridge resolve a mixed
# set of same-process ACLNN symbols, so a standalone build is deliberately
# forbidden. Use ../build_p10_target_assembly.sh in build mode instead.

set -euo pipefail

printf '%s\n' \
  'P10_FORWARD_PREPARE_STANDALONE_BLOCKED reason=combined_product_vendor_required use=build_p10_target_assembly.sh' \
  >&2
exit 2
