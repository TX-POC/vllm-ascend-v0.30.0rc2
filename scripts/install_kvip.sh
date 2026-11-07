#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the external operators; never copy their source into vLLM.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kvip_dir="$repo_dir/third_party/KVIP"
revision=c5b17ef29287483a65e9d7d4ffe3ce7ac33702ff
kvaf_revision=1c1310387ebfb6776a07316874e924265b209ada
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
if ! git -C "$kvip_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$kvip_dir" fetch origin "$revision"
fi
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
# Prepare upstream patches first; our Put binding change releases the GIL only
# around native execution. The synchronous API and device algorithm are unchanged.
bash "$kvip_dir/scripts/build.sh" --kvaf-root "$repo_dir/third_party/KVAF" --prepare-only
put_patch="$repo_dir/scripts/patches/kvaf-background-put.patch"
if ! git -C "$repo_dir/third_party/KVAF" apply --reverse --check "$put_patch" 2>/dev/null; then
    git -C "$repo_dir/third_party/KVAF" apply --check "$put_patch"
    git -C "$repo_dir/third_party/KVAF" apply "$put_patch"
fi
# KVIP owns operator builds and Python installation.
bash "$kvip_dir/scripts/build.sh" --kvaf-root "$repo_dir/third_party/KVAF" ${build_args[@]+"${build_args[@]}"}
echo "KVIP ready. Source $kvip_dir/scripts/env.sh before standalone checks."
