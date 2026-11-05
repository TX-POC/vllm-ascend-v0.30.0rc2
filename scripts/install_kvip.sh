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
# KVIP is a disposable build dependency: recreate it at the pinned revision.
if [[ -e "$kvip_dir" || -L "$kvip_dir" ]]; then
    echo "Removing existing KVIP source/build (including local edits): $kvip_dir"
    rm -rf -- "$kvip_dir"
fi
mkdir -p "$repo_dir/third_party"
git clone --branch main https://gitcode.com/CacheOS/KVIP.git "$kvip_dir"
abc_patch="$repo_dir/scripts/patches/kvip-configurable-abc.patch"
if ! git -C "$kvip_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$kvip_dir" fetch origin main
fi
git -C "$kvip_dir" checkout --detach "$revision"
git -C "$kvip_dir" apply "$abc_patch"
git -C "$kvip_dir" apply "$repo_dir/scripts/patches/kvip-batched-page-updates.patch"
echo "KVIP source ready: $kvip_dir @ $revision (git-ignored external dependency)"
if [[ "$source_only" == true ]]; then
    exit 0
fi
# Prepare upstream patches first; our Put binding change releases the GIL only
# around native execution. The synchronous API and device algorithm are unchanged.
bash "$kvip_dir/scripts/build.sh" --kvaf-root "$repo_dir/third_party/KVAF" --prepare-only
put_patch="$repo_dir/scripts/patches/kvaf-background-put.patch"
if ! git -C "$repo_dir/third_party/KVAF" apply --reverse --check "$put_patch" 2>/dev/null; then
    git -C "$repo_dir/third_party/KVAF" apply --check "$put_patch"
    git -C "$repo_dir/third_party/KVAF" apply "$put_patch"
fi
# KVIP owns operator builds and Python installation.
bash "$kvip_dir/scripts/build.sh" --kvaf-root "$repo_dir/third_party/KVAF" "${build_args[@]}"
echo "KVIP ready. Source $kvip_dir/scripts/env.sh before standalone checks."
