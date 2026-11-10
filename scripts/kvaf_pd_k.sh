#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 build KVAF_SOURCE BUILD_DIR [cmake options...] | start BUILD_DIR LISTEN_IP [service options...]" >&2
    exit 2
fi
action=$1
shift
case "$action" in
    build)
        source_dir=$(realpath "$1")
        build_dir=$(realpath -m "$2")
        shift 2
        test -f "$source_dir/docs/connector_pull_api_design.md"
        cmake -S "$source_dir" -B "$build_dir" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_BUILD_PYTHON=OFF \
            -DKVAF_ENABLE_CPU_OFFLOAD=OFF -DKVAF_ENABLE_AIV_BATCH_GET=OFF \
            -DKVAF_ENABLE_AIV_BATCH_PUT=OFF -DBUILD_TESTING=OFF "$@"
        cmake --build "$build_dir" -j "${KVAF_PD_BUILD_JOBS:-8}"
        ;;
    start)
        build_dir=$(realpath "$1")
        listen_ip=$2
        shift 2
        exec "$build_dir/runtime/kvaf_cpu_service" --listen "$listen_ip" --memory-page-mode default "$@"
        ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
