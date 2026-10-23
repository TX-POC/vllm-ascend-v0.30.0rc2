#!/usr/bin/env bash
# Based on the verified A5 container command (image 6aba788b878c).
# Usage: bash scripts/start_vllm_container.sh IMAGE [CONTAINER_NAME]
# Requires an A5 host and a v0.30.0 development image built with Dockerfile.a5
# or Dockerfile.a5.openEuler (including /vllm-workspace/vllm and build deps).

set -euo pipefail

image="${1:?Usage: $0 IMAGE [CONTAINER_NAME]}"
container_name="${2:-vllm-ascend-v0.30.0rc2}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

device_ids="${ASCEND_VISIBLE_DEVICES:-}"
if [[ -z "$device_ids" ]]; then
    for device in /dev/davinci[0-9]*; do
        [[ -c "$device" ]] || continue
        device_ids+="${device_ids:+,}${device#/dev/davinci}"
    done
fi
if [[ ! "$device_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo 'No valid NPU selection. Set ASCEND_VISIBLE_DEVICES or check /dev/davinci*.' >&2
    exit 1
fi
IFS=',' read -r -a devices <<< "$device_ids"
docker_args=(--net=host --shm-size=512g --privileged=true --runtime=runc
    --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/ummu --device=/dev/uburma)
for device in "${devices[@]}"; do
    docker_args+=("--device=/dev/davinci$device")
done
# Let Docker copy exported proxy values without putting credentials in argv.
for proxy_name in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy; do
    if [[ -n "${!proxy_name:-}" ]]; then
        docker_args+=(--env "$proxy_name")
    fi
done

# Match the verified command's IPC, init process, devices and writable mounts.
# Docker refuses an existing container name; no container is removed here.
printf 'Creating %s with host NPU devices: %s\n' "$container_name" "$device_ids"
docker run -it -d --name "$container_name" -u root \
    "${docker_args[@]}" \
    --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver \
    --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi \
    --mount type=bind,src=/usr/local/sbin,dst=/usr/local/sbin \
    --mount type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi \
    --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info \
    --mount type=bind,src=/etc/hccl_rootinfo.json,dst=/etc/hccl_rootinfo.json \
    --mount type=bind,src=/etc/hixlep,dst=/etc/hixlep \
    --mount type=bind,src=/root/host,dst=/root/host \
    --mount type=bind,src=/var/log/npu,dst=/usr/slog \
    --mount type=bind,src=/home,dst=/home \
    --mount type=bind,src=/mnt,dst=/mnt \
    --mount type=bind,src=/data,dst=/data \
    --mount "type=bind,src=$repo_root,dst=/vllm-workspace/vllm-ascend" \
    -e SOC_VERSION=ascend950dt_9582 -e COMPILE_CUSTOM_KERNELS=1 \
    -w /vllm-workspace/vllm-ascend "$image" /usr/sbin/init

printf 'Ready. Enter with: docker exec -it %q bash\n' "$container_name"
printf 'Then install inside the container: bash /vllm-workspace/vllm-ascend/scripts/install_vllm_in_container.sh\n'
