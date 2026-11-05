#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compatibility entry: KVIP now owns preparation/build of the shared KVAF dependency.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--fused-context" ) ]]; then
    echo "Usage: bash scripts/install_kvaf.sh [--fused-context]" >&2
    exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_dir/scripts/install_kvip.sh"
