#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Edit these example addresses and topology before deployment.
P_HOSTS=(141.62.38.12 141.62.38.13)
D_HOSTS=(141.62.38.24 141.62.38.25)
K_HOSTS=(141.62.38.26 141.62.38.27 141.62.38.28 141.62.38.29 141.62.38.30 141.62.38.31 141.62.38.32 141.62.38.33)
K_PORTS=(28997 28999)
# Each entry must cover one complete CPU socket; verify with lscpu/numactl.
K_NUMA_NODES=(0 1)
NIC=enp34s0f1
REPO=/home/c00612283/vllm-ascend-v0.30.0rc2
SDK=/home/c00612283/KVAF-Devlop
BUILD=$SDK/build-pd-py312
K_BUILD=$SDK/build-hostcpu-k
PYTHON=/usr/local/python3.12.13/bin/python3
HOST_PYTHON=python3
MODEL=/home/models/GLM-5.3-w4a4c8-mxfp4
LOG_ROOT=/home/c00612283/pd-pool-logs
COORDINATOR=${K_HOSTS[0]}:28998
CONTROL_BACKLOG=128  # D framework listener queue; not a Jetty/thread limit.
PROFILER_DIR=/home/d00843668/profiling_no_pooling
DIAGNOSTICS_REQUESTS=1
VERIFY_KV=true
