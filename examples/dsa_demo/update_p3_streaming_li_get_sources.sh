#!/usr/bin/env bash
# Synchronize the P3 streaming-LI/Get prototype through Gitcode only.
#
# Run this *inside an A5 Docker container* as:
#   bash examples/dsa_demo/update_p3_streaming_li_get_sources.sh
#
# It is intentionally a source-update/check utility.  It never builds,
# deletes, resets, stashes, or changes Git configuration.  A non-empty
# tracked worktree is left untouched and reported as blocked.

P3_VLLM_BRANCH="${P3_VLLM_BRANCH:-kip_demo_cs}"
P3_KVAF_BRANCH="${P3_KVAF_BRANCH:-kip_demo_cs_native_enqueue}"
# Pin both SHAs in a reproducible validation run.  When they are omitted, the
# script resolves the currently fetched origin/<branch> tip and prints it.
P3_VLLM_SHA="${P3_VLLM_SHA:-}"
P3_KVAF_SHA="${P3_KVAF_SHA:-}"

mode="${1:-update}"

print_usage() {
  printf '%s\n' \
    'usage: bash update_p3_streaming_li_get_sources.sh [update|check]' \
    'optional overrides: KIP_ROOT, VLLM_KIP, KVAF_FUSED,' \
    'P3_VLLM_BRANCH, P3_KVAF_BRANCH, P3_VLLM_SHA, P3_KVAF_SHA'
}

detect_paths() {
  if [ -z "${KIP_ROOT:-}" ]; then
    for candidate in /data/cacheos_kip_demo_cs /home/c00982601; do
      if [ -e "${candidate}/vLLM-DSA-kip_demo_cs/.git" ] && \
         [ -e "${candidate}/KVAF-kip_demo_cs_native_enqueue/.git" ]; then
        KIP_ROOT="${candidate}"
        break
      fi
    done
  fi

  if [ -n "${KIP_ROOT:-}" ]; then
    : "${VLLM_KIP:=${KIP_ROOT}/vLLM-DSA-kip_demo_cs}"
    : "${KVAF_FUSED:=${KIP_ROOT}/KVAF-kip_demo_cs_native_enqueue}"
  fi

  if [ ! -e "${VLLM_KIP:-}/.git" ] || [ ! -e "${KVAF_FUSED:-}/.git" ]; then
    printf 'P3_UPDATE_BLOCKED reason=missing_worktree kip_root=<%s> vllm=<%s> kvaf=<%s>\n' \
      "${KIP_ROOT:-}" "${VLLM_KIP:-}" "${KVAF_FUSED:-}"
    return 1
  fi
  return 0
}

repo_state() {
  local name="$1"
  local path="$2"
  local branch="$3"
  local expected="$4"
  local dirty
  local head

  if [ -z "${expected}" ]; then
    expected="$(git -C "${path}" rev-parse "origin/${branch}" 2>/dev/null || true)"
  fi
  if [ -z "${expected}" ]; then
    expected="<origin-unfetched>"
  fi

  dirty="$(git -C "${path}" status --porcelain --untracked-files=no 2>/dev/null)"
  head="$(git -C "${path}" rev-parse HEAD 2>/dev/null || true)"
  printf 'P3_UPDATE_STATE repo=%s path=%s branch=%s head=%s expected=%s tracked_dirty=%s\n' \
    "${name}" "${path}" "${branch}" "${head}" "${expected}" \
    "$([ -n "${dirty}" ] && echo 1 || echo 0)"
}

update_repo() {
  local name="$1"
  local path="$2"
  local branch="$3"
  local expected="$4"
  local dirty
  local fetch_rc
  local merge_rc

  if [ -z "${expected}" ]; then
    expected="<branch-tip-after-fetch>"
  fi

  dirty="$(git -C "${path}" status --porcelain --untracked-files=no 2>/dev/null)"
  if [ -n "${dirty}" ]; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=tracked_changes\n' "${name}"
    git -C "${path}" status --short
    return 1
  fi

  printf 'P3_UPDATE_FETCH repo=%s branch=%s\n' "${name}" "${branch}"
  # Git owns any interactive Username/Password prompt.  Each repository is
  # fetched in a separate command so credentials cannot be mixed.
  git -C "${path}" fetch origin "${branch}"
  fetch_rc=$?
  if [ "${fetch_rc}" -ne 0 ]; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=fetch_failed status=%s\n' \
      "${name}" "${fetch_rc}"
    return 1
  fi

  git -C "${path}" merge --ff-only "origin/${branch}"
  merge_rc=$?
  if [ "${merge_rc}" -ne 0 ]; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=ff_only_merge_failed status=%s\n' \
      "${name}" "${merge_rc}"
    return 1
  fi

  repo_state "${name}" "${path}" "${branch}" "${expected}"
  return 0
}

main() {
  local overall=0
  local vllm_head
  local kvaf_head
  local vllm_expected
  local kvaf_expected

  case "${mode}" in
    update|check) ;;
    *)
      print_usage
      return 1
      ;;
  esac

  if ! detect_paths; then
    return 1
  fi

  printf 'P3_UPDATE_HOST hostname=%s kip_root=%s mode=%s\n' \
    "$(hostname)" "${KIP_ROOT:-override}" "${mode}"
  repo_state vllm "${VLLM_KIP}" "${P3_VLLM_BRANCH}" "${P3_VLLM_SHA}"
  repo_state kvaf "${KVAF_FUSED}" "${P3_KVAF_BRANCH}" "${P3_KVAF_SHA}"

  if [ "${mode}" = "update" ]; then
    update_repo vllm "${VLLM_KIP}" "${P3_VLLM_BRANCH}" "${P3_VLLM_SHA}" || overall=1
    update_repo kvaf "${KVAF_FUSED}" "${P3_KVAF_BRANCH}" "${P3_KVAF_SHA}" || overall=1
  fi

  vllm_head="$(git -C "${VLLM_KIP}" rev-parse HEAD 2>/dev/null || true)"
  kvaf_head="$(git -C "${KVAF_FUSED}" rev-parse HEAD 2>/dev/null || true)"
  vllm_expected="${P3_VLLM_SHA:-$(git -C "${VLLM_KIP}" rev-parse "origin/${P3_VLLM_BRANCH}" 2>/dev/null || true)}"
  kvaf_expected="${P3_KVAF_SHA:-$(git -C "${KVAF_FUSED}" rev-parse "origin/${P3_KVAF_BRANCH}" 2>/dev/null || true)}"
  printf 'P3_UPDATE_FINAL vllm=%s expected=%s kvaf=%s expected=%s ready=%s\n' \
    "${vllm_head}" "${vllm_expected}" \
    "${kvaf_head}" "${kvaf_expected}" \
    "$([ "${overall}" -eq 0 ] && [ -n "${vllm_expected}" ] && \
       [ -n "${kvaf_expected}" ] && [ "${vllm_head}" = "${vllm_expected}" ] && \
       [ "${kvaf_head}" = "${kvaf_expected}" ] && echo 1 || echo 0)"
  return "${overall}"
}

main "$@"
