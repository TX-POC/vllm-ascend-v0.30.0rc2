#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the external operators; never copy their source into vLLM.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kvip_dir="$repo_dir/third_party/KVIP"
revision=81b1472809b7a5f3469e26467feee0ca1c61d71a
branch=main
kvaf_revision=1c1310387ebfb6776a07316874e924265b209ada
source_only=false
skip_kvaf=false
build_args=()
for argument in "$@"; do
    case "$argument" in
        --source-only) source_only=true ;;
        --transport=urma|--transport=ubmem) ;; # Accepted for compatibility; always build both.
        --skip-kvaf-build) skip_kvaf=true; build_args+=("$argument") ;;
        --help|-h)
            echo "Usage: bash scripts/install_kvip.sh [--source-only] [--skip-kvaf-build]"
            exit 0 ;;
        *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
done
# Do not reuse a URMA-only or unknown KVAF binary for a dual-transport install.
if [[ "$skip_kvaf" == true ]]; then
    if [[ ! -f "$repo_dir/third_party/KVAF/access_operator/include/ubmem/fused_batch_get_device.h" ]] ||
        ! python3 - "$kvip_dir/build/dependencies.json" "$repo_dir/third_party/KVAF/build-npu" <<'PYSTATE'
import json
import sys
from pathlib import Path
try:
    state = json.loads(Path(sys.argv[1]).read_text())
    valid = (state.get("transport") == "ubmem" and
             Path(state["kvaf_build_dir"]).resolve() == Path(sys.argv[2]).resolve())
except (OSError, ValueError, KeyError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PYSTATE
    then
        echo "Previous UBmem KVAF build not verified; rebuilding KVAF."
        build_args=()
    fi
fi
# Preserve any existing checkout, including local edits and build outputs.
mkdir -p "$repo_dir/third_party"
if [[ -e "$kvip_dir" || -L "$kvip_dir" ]]; then
    backup="$(mktemp -d "$repo_dir/third_party/KVIP-backup.XXXXXX")"
    mv "$kvip_dir" "$backup/KVIP"
    echo "Previous KVIP preserved: $backup/KVIP"
fi
git clone --branch "$branch" https://gitcode.com/CacheOS/KVIP.git "$kvip_dir"
git -C "$kvip_dir" checkout --detach "$revision"
echo "KVIP source ready: $kvip_dir @ $revision (git-ignored external dependency)"
if [[ "$source_only" == true ]]; then
    exit 0
fi
# Preserve a different checkout (including local patches/builds) before switching versions.
kvaf_dir="$repo_dir/third_party/KVAF"
if [[ -e "$kvaf_dir" ]] && {
    [[ "$(git -C "$kvaf_dir" rev-parse HEAD)" != "$kvaf_revision" ]] ||
    grep -q 'kvaf_build_isolated_urma_get' "$kvaf_dir/access_operator/demo/aiv_batch_get/kernel/CMakeLists.txt" 2>/dev/null
}; then
    backup="$(mktemp -d "$repo_dir/third_party/KVAF-backup.XXXXXX")"
    mv "$kvaf_dir" "$backup/KVAF"
    echo "Previous KVAF preserved: $backup/KVAF"
    # A changed SDK must be rebuilt even if --skip-kvaf-build was supplied.
    build_args=()
fi
# The UBmem KVAF build includes URMA. Build UBmem first, then warm the
# original URMA LIFetch cache against exactly the same KVAF checkout.
bash "$kvip_dir/scripts/build.sh" --kvaf-root "$kvaf_dir" --transport ubmem ${build_args[@]+"${build_args[@]}"}
PYTHONPATH="$kvip_dir/python${PYTHONPATH:+:$PYTHONPATH}" python3 - "$kvaf_dir/build-npu" <<'PYBUILD'
import sys
from pathlib import Path
from kvip.runtime.build import build_libraries, kvaf_build_paths

build_dir = Path(sys.argv[1])
cann, headers = kvaf_build_paths(build_dir)
paths = build_libraries(cache_dir=build_dir / "lifetch", cann=cann, headers=headers, soc="Ascend950PR_9599")
print(f"URMA LIFetch ready: {paths}", flush=True)
PYBUILD
echo "KVIP ready for URMA and UBmem. Source $kvip_dir/scripts/env.sh before standalone checks."
