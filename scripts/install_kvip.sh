#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the external operators; never copy their source into vLLM.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kvip_dir="$repo_dir/third_party/KVIP"
kvip_branch=main
kvip_revision=d6cfd392f722dd3f96ef6e05175f503aac511dbb
kvaf_branch=main
kvaf_revision=813f1750a50b7e6c92d5f5397f2fbc05befef6da
source_only=false
build_args=()
main_kvip_dir=""
main_kvaf_dir=""
while (( $# )); do
    case "$1" in
        --source-only) source_only=true ;;
        --main-kvaf)
            if (( $# < 3 )) || [[ ! -d "$2" || ! -d "$3" ]]; then
                echo "--main-kvaf requires existing KVIP and KVAF checkout directories" >&2
                exit 2
            fi
            main_kvip_dir="$(cd "$2" && pwd)"
            main_kvaf_dir="$(cd "$3" && pwd)"
            shift 2 ;;

        --transport)
            case "${2:-}" in
                urma|ubmem) shift ;;
                *) echo "--transport requires urma or ubmem" >&2; exit 2 ;;
            esac ;;
        --transport=urma|--transport=ubmem) ;; # Accepted for compatibility; always build both.
        --skip-kvaf-build) build_args+=("$1") ;;
        --help|-h)
            echo "Usage: bash scripts/install_kvip.sh [--source-only] [--skip-kvaf-build]"
            echo "       bash scripts/install_kvip.sh --main-kvaf KVIP_CHECKOUT KVAF_CHECKOUT [--source-only] [--skip-kvaf-build]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
if [[ -n "$main_kvaf_dir" ]]; then
    kvip_dir="$main_kvip_dir"
    kvaf_dir="$main_kvaf_dir"
    if [[ "$(git -C "$kvip_dir" rev-parse HEAD)" != "$kvip_revision" ||
          "$(git -C "$kvaf_dir" rev-parse HEAD)" != "$kvaf_revision" ]]; then
        echo "Expected KVIP $kvip_revision and KVAF $kvaf_revision" >&2
        exit 1
    fi
else
    kvaf_dir="$repo_dir/third_party/KVAF"
    for dependency in KVIP KVAF; do
        if [[ "$dependency" == KVIP ]]; then
            directory="$kvip_dir"
            branch="$kvip_branch"
            revision="$kvip_revision"
            repository=https://github.com/TX-POC/KVAF-KVIP.git
        else
            directory="$kvaf_dir"
            branch="$kvaf_branch"
            revision="$kvaf_revision"
            repository=https://github.com/TX-POC/KVAF-Devlop.git
        fi
        if [[ -d "$directory/.git" || -f "$directory/.git" ]] &&
           [[ "$(git -C "$directory" rev-parse HEAD)" == "$revision" ]]; then
            echo "$dependency source ready: $directory @ $revision"
            continue
        fi
        if [[ -e "$directory" || -L "$directory" ]]; then
            backup="$(mktemp -d "$repo_dir/third_party/$dependency-backup.XXXXXX")"
            mv "$directory" "$backup/$dependency"
            echo "Previous $dependency preserved: $backup/$dependency"
        fi
        git clone --no-checkout --branch "$branch" "$repository" "$directory"
        git -C "$directory" checkout --detach "$revision"
        echo "$dependency source ready: $directory @ $revision"
        if [[ "$dependency" == KVAF ]]; then
            build_args=()
        fi
    done
fi
if [[ "$source_only" == true ]]; then
    exec bash "$kvip_dir/scripts/build.sh" --prepare-only --kvaf-root "$kvaf_dir" --transport ubmem
fi
cann_args=()
if [[ -n "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
    cann_args=(--cann "$ASCEND_CANN_PACKAGE_PATH")
fi
bash "$kvip_dir/scripts/build.sh" --kvaf-root "$kvaf_dir" --transport ubmem "${cann_args[@]}" "${build_args[@]}"
PYTHONPATH="$kvip_dir/python${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" - "$kvaf_dir/build-npu" <<'PYBUILD'
import sys
from pathlib import Path
from kvip.runtime.build import build_libraries, kvaf_build_paths

build_dir = Path(sys.argv[1])
cann, headers = kvaf_build_paths(build_dir)
paths = build_libraries(cache_dir=build_dir / "lifetch", cann=cann, headers=headers, soc="Ascend950PR_9599")
print(f"URMA LIFetch ready: {paths}", flush=True)
PYBUILD
echo "KVIP ready for URMA and UBmem. Source $kvip_dir/scripts/env.sh before standalone checks."
