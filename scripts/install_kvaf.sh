#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kvaf_revision="c4375e13d7a9519d907ab6a5d11227795f207286"
kvaf_repository="https://gitcode.com/CacheOS/KVAF.git"
source_dir="$repo_root/third_party/KVAF"

if [[ ! -e "$source_dir" ]]; then
    mkdir -p "$(dirname "$source_dir")"
    git clone --no-checkout "$kvaf_repository" "$source_dir"
    git -C "$source_dir" checkout --detach "$kvaf_revision"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$kvaf_revision" ]]; then
    echo "KVAF expects revision $kvaf_revision; existing checkout was preserved: $source_dir" >&2
    exit 1
fi

# Apply the integration fix once; preserve unrelated edits in the KVAF checkout.
patch_file="$repo_root/scripts/patches/kvaf-reuse-device-ra.patch"
if ! git -C "$source_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    git -C "$source_dir" apply --check "$patch_file"
    git -C "$source_dir" apply "$patch_file"
fi

# Keep automatically downloaded CPU dependencies inside the ignored KVAF tree.
KVAF_UMDK_ROOT="${KVAF_UMDK_ROOT:-$source_dir/third_party/umdk}" \
    bash "$source_dir/build.sh" cpu
bash "$source_dir/build.sh" npu

# Register the binding in the interpreter used by KVAF's build.sh.
"${PYTHON:-python3}" - "$source_dir/build-npu/python" <<'PY'
import pathlib
import sys
import sysconfig

pth = pathlib.Path(sysconfig.get_path("purelib")) / "vllm_ascend_kvaf.pth"
pth.write_text(sys.argv[1] + "\n", encoding="utf-8")
print(f"Registered KVAF for {sys.executable}: {pth}")
PY
echo "KVAF CPU service and NPU kernels ready: $source_dir ($kvaf_revision)"
