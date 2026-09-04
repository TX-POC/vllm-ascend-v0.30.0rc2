#!/usr/bin/env bash
# Materialize the CANN build dependencies into one external, reusable cache.
#
# This intentionally does not use a VLLM Git worktree: csrc/third_party is
# ignored by Git, so a clean clone is not a complete CANN dependency cache.
# The caller may override every URL for an internal mirror.  Credentials are
# never read from Git configuration; curl may use the normal proxy/netrc setup
# of the validation container when available.

set -euo pipefail

cache_root="${1:-${P10_CANN_3RD_LIB_PATH:-}}"

fail() {
    printf 'P10_CANN_CACHE_BOOTSTRAP_FAIL reason=%s\n' "$1" >&2
    exit 2
}

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

has_makeself() {
    [[ -r "${cache_root}/makeself/makeself.sh" ]] &&
        [[ -r "${cache_root}/makeself/makeself-header.sh" ]]
}

has_json() {
    [[ -r "${cache_root}/json/include/nlohmann/json.hpp" ]]
}

has_abseil() {
    [[ -r "${cache_root}/abseil-cpp/abseil-cpp-20230802.1.tar.gz" ]] ||
        [[ -r "${cache_root}/pkg/abseil-cpp-20230802.1.tar.gz" ]]
}

abseil_archive() {
    if [[ -r "${cache_root}/abseil-cpp/abseil-cpp-20230802.1.tar.gz" ]]; then
        printf '%s\n' "${cache_root}/abseil-cpp/abseil-cpp-20230802.1.tar.gz"
    else
        printf '%s\n' "${cache_root}/pkg/abseil-cpp-20230802.1.tar.gz"
    fi
}

has_protobuf() {
    [[ -r "${cache_root}/protobuf/protobuf-all-25.1.tar.gz" ]] ||
        [[ -r "${cache_root}/pkg/protobuf-25.1.tar.gz" ]]
}

protobuf_archive() {
    if [[ -r "${cache_root}/protobuf/protobuf-all-25.1.tar.gz" ]]; then
        printf '%s\n' "${cache_root}/protobuf/protobuf-all-25.1.tar.gz"
    else
        printf '%s\n' "${cache_root}/pkg/protobuf-25.1.tar.gz"
    fi
}

cache_complete() {
    has_makeself && has_json && has_abseil && has_protobuf
}

[[ -n "${cache_root}" ]] || fail "cache_path_missing"
mkdir -p "${cache_root}" || fail "cache_create_failed:path=${cache_root}"
cache_root="$(cd "${cache_root}" && pwd -P)"

if cache_complete; then
    printf 'P10_CANN_CACHE_BOOTSTRAP_PASS path=%s reused=1\n' "${cache_root}"
    exit 0
fi

makeself_missing=0
json_missing=0
abseil_missing=0
protobuf_missing=0
has_makeself || makeself_missing=1
has_json || json_missing=1
has_abseil || abseil_missing=1
has_protobuf || protobuf_missing=1

command -v curl >/dev/null 2>&1 || fail "curl_missing"
if [[ "${makeself_missing}" -eq 1 || "${abseil_missing}" -eq 1 ||
      "${protobuf_missing}" -eq 1 ]]; then
    command -v tar >/dev/null 2>&1 || fail "tar_missing"
fi
if [[ "${json_missing}" -eq 1 ]]; then
    command -v unzip >/dev/null 2>&1 || fail "unzip_missing"
fi

makeself_url="${P10_CANN_3RD_MAKESELF_URL:-https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz}"
json_url="${P10_CANN_3RD_JSON_URL:-https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip}"
abseil_url="${P10_CANN_3RD_ABSEIL_URL:-https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz}"
protobuf_url="${P10_CANN_3RD_PROTOBUF_URL:-https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz}"

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/p10-cann-cache.XXXXXX")"
cleanup() { rm -rf "${tmp_root}"; }
trap cleanup EXIT

download() {
    local label="$1"
    local url="$2"
    local output="$3"
    local -a args=(--fail --location --retry 3 --retry-delay 2 --connect-timeout 15 --max-time 900 --netrc-optional)
    if [[ -n "${P10_CANN_3RD_CURL_CONFIG:-}" ]]; then
        [[ -r "${P10_CANN_3RD_CURL_CONFIG}" ]] ||
            fail "curl_config_unreadable:path=${P10_CANN_3RD_CURL_CONFIG}"
        args+=(--config "${P10_CANN_3RD_CURL_CONFIG}")
    fi
    printf 'P10_CANN_CACHE_DOWNLOAD label=%s url=%s\n' "${label}" "${url}"
    curl "${args[@]}" --output "${output}" "${url}" ||
        fail "download_failed:label=${label}:url=${url}"
    [[ -s "${output}" ]] || fail "download_empty:label=${label}"
}

mkdir -p "${cache_root}/makeself" "${cache_root}/json/include" "${cache_root}/pkg"

if [[ "${makeself_missing}" -eq 1 ]]; then
    download makeself "${makeself_url}" "${tmp_root}/makeself.tar.gz"
    tar -tzf "${tmp_root}/makeself.tar.gz" >/dev/null || fail "makeself_archive_invalid"
    mkdir -p "${tmp_root}/makeself"
    tar -xzf "${tmp_root}/makeself.tar.gz" -C "${tmp_root}/makeself"
    makeself_bin="$(find "${tmp_root}/makeself" -type f -name makeself.sh -print -quit)"
    [[ -n "${makeself_bin}" ]] || fail "makeself_binary_missing_after_extract"
    makeself_dir="$(dirname "${makeself_bin}")"
    [[ -r "${makeself_dir}/makeself-header.sh" ]] ||
        fail "makeself_header_missing_after_extract"
    install -m 0755 "${makeself_dir}/makeself.sh" "${cache_root}/makeself/makeself.sh"
    install -m 0644 "${makeself_dir}/makeself-header.sh" \
        "${cache_root}/makeself/makeself-header.sh"
fi

if [[ "${json_missing}" -eq 1 ]]; then
    download json "${json_url}" "${tmp_root}/json.zip"
    unzip -tq "${tmp_root}/json.zip" >/dev/null || fail "json_archive_invalid"
    mkdir -p "${tmp_root}/json"
    unzip -q "${tmp_root}/json.zip" -d "${tmp_root}/json"
    json_header="$(find "${tmp_root}/json" -type f -path '*/nlohmann/json.hpp' -print -quit)"
    [[ -n "${json_header}" ]] || fail "json_header_missing_after_extract"
    json_include="$(dirname "$(dirname "${json_header}")")"
    rm -rf "${cache_root}/json/include/nlohmann"
    cp -a "${json_include}/nlohmann" "${cache_root}/json/include/"
fi

if [[ "${abseil_missing}" -eq 1 ]]; then
    download abseil "${abseil_url}" "${tmp_root}/abseil.tar.gz"
    tar -tzf "${tmp_root}/abseil.tar.gz" >/dev/null || fail "abseil_archive_invalid"
    install -m 0644 "${tmp_root}/abseil.tar.gz" \
        "${cache_root}/pkg/abseil-cpp-20230802.1.tar.gz"
fi

if [[ "${protobuf_missing}" -eq 1 ]]; then
    download protobuf "${protobuf_url}" "${tmp_root}/protobuf.tar.gz"
    tar -tzf "${tmp_root}/protobuf.tar.gz" >/dev/null || fail "protobuf_archive_invalid"
    install -m 0644 "${tmp_root}/protobuf.tar.gz" \
        "${cache_root}/pkg/protobuf-25.1.tar.gz"
fi

cache_complete || fail "cache_incomplete_after_download:path=${cache_root}"
{
    printf 'format=1\n'
    printf 'kind=p10_cann_third_party_cache\n'
    printf 'makeself_action=%s\n' "$([[ "${makeself_missing}" -eq 1 ]] && echo downloaded || echo reused)"
    printf 'json_action=%s\n' "$([[ "${json_missing}" -eq 1 ]] && echo downloaded || echo reused)"
    printf 'abseil_action=%s\n' "$([[ "${abseil_missing}" -eq 1 ]] && echo downloaded || echo reused)"
    printf 'protobuf_action=%s\n' "$([[ "${protobuf_missing}" -eq 1 ]] && echo downloaded || echo reused)"
    printf 'makeself_url=%s\n' "${makeself_url}"
    printf 'json_url=%s\n' "${json_url}"
    printf 'abseil_url=%s\n' "${abseil_url}"
    printf 'protobuf_url=%s\n' "${protobuf_url}"
    printf 'makeself_sha256=%s\n' "$(hash_file "${cache_root}/makeself/makeself.sh")"
    printf 'json_sha256=%s\n' "$(hash_file "${cache_root}/json/include/nlohmann/json.hpp")"
    printf 'abseil_sha256=%s\n' "$(hash_file "$(abseil_archive)")"
    printf 'protobuf_sha256=%s\n' "$(hash_file "$(protobuf_archive)")"
} > "${cache_root}/P10_CANN_3RD_PROVENANCE"

printf 'P10_CANN_CACHE_BOOTSTRAP_PASS path=%s reused=0 manifest=%s\n' \
    "${cache_root}" "${cache_root}/P10_CANN_3RD_PROVENANCE"
