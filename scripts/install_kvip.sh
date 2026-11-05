#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the external operators; never copy their source into vLLM.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kvip_dir="$repo_dir/third_party/KVIP"
revision=46c03fcf889a3ab56e62c1dbf5acb60580120142
source_only=false
build_args=()
for argument in "$@"; do
    case "$argument" in
        --source-only) source_only=true ;;
        --skip-kvaf-build) build_args+=("$argument") ;;
        --help|-h)
            echo "Usage: bash scripts/install_kvip.sh [--source-only] [--skip-kvaf-build]"
            exit 0 ;;
        *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
done
if [[ ! -e "$kvip_dir" ]]; then
    mkdir -p "$repo_dir/third_party"
    git clone --branch main https://gitcode.com/CacheOS/KVIP.git "$kvip_dir"
fi
# Do not reset experiments or patches in an existing checkout.
if [[ -n "$(git -C "$kvip_dir" status --porcelain)" ]]; then
    echo "KVIP checkout has local changes; preserve/move it before installing: $kvip_dir" >&2
    exit 1
fi
if ! git -C "$kvip_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$kvip_dir" fetch origin main
fi
git -C "$kvip_dir" checkout --detach "$revision"
echo "KVIP source ready: $kvip_dir @ $revision (git-ignored external dependency)"
if [[ "$source_only" == true ]]; then
    exit 0
fi
# KVIP owns its KVAF patches, operator builds and Python installation.
bash "$kvip_dir/scripts/build.sh" --kvaf-root "$repo_dir/third_party/KVAF" "${build_args[@]}"
echo "KVIP ready. Source $kvip_dir/scripts/env.sh before standalone checks."
