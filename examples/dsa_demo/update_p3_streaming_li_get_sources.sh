#!/usr/bin/env bash
# Synchronize the P3 streaming-LI/Get prototype through Gitcode only.
#
# Run this *inside an A5 Docker container* as:
#   P3_VLLM_SHA=<full-sha> P3_KVAF_SHA=<full-sha> \
#     bash examples/dsa_demo/update_p3_streaming_li_get_sources.sh update
#
# The two pinned SHAs are deliberately mandatory: a remote test must consume
# exactly the source revision published by the local developer, not whichever
# branch tip happens to exist at execution time.
#
# This utility never builds, deletes, resets, stashes, clones, or changes Git
# configuration. It first fetches and preflights *both* repositories, then
# fast-forwards both to their exact pinned SHAs. Detached worktrees are valid;
# a named checkout on an unexpected branch is reported and left untouched.

P3_VLLM_BRANCH="${P3_VLLM_BRANCH:-kip_demo_cs}"
P3_KVAF_BRANCH="${P3_KVAF_BRANCH:-kip_demo_cs_native_enqueue}"
P3_VLLM_SHA="${P3_VLLM_SHA:-}"
P3_KVAF_SHA="${P3_KVAF_SHA:-}"
mode="${1:-update}"

print_usage() {
  printf '%s\n' \
    'usage: P3_VLLM_SHA=<full-sha> P3_KVAF_SHA=<full-sha> bash update_p3_streaming_li_get_sources.sh [update|check]' \
    'optional path overrides: KIP_ROOT, VLLM_KIP, KVAF_FUSED' \
    'optional branch overrides: P3_VLLM_BRANCH, P3_KVAF_BRANCH'
}

tracked_count() {
  git -C "$1" status --porcelain --untracked-files=no 2>/dev/null \
    | awk 'NF { count += 1 } END { print count + 0 }'
}

checkout_name() {
  local name
  name="$(git -C "$1" branch --show-current 2>/dev/null)"
  if [ -n "$name" ]; then
    printf '%s' "$name"
  else
    printf '%s' '<detached>'
  fi
}

detect_paths() {
  local candidate

  if [ -z "${KIP_ROOT:-}" ]; then
    for candidate in /data/cacheos_kip_demo_cs /home/c00982601; do
      if [ -e "${candidate}/vLLM-DSA-kip_demo_cs/.git" ] && \
         [ -e "${candidate}/KVAF-kip_demo_cs_native_enqueue/.git" ]; then
        KIP_ROOT="$candidate"
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

print_state() {
  local repo="$1"
  local path="$2"
  local target_branch="$3"
  local expected="$4"
  local head
  local branch
  local dirty

  head="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
  branch="$(checkout_name "$path")"
  dirty="$(tracked_count "$path")"
  printf 'P3_UPDATE_STATE repo=%s path=%s checkout=%s target_branch=%s head=%s expected=%s tracked_dirty=%s\n' \
    "$repo" "$path" "$branch" "$target_branch" "$head" "$expected" "$dirty"
}

precheck_checkout() {
  local repo="$1"
  local path="$2"
  local target_branch="$3"
  local branch
  local dirty

  branch="$(git -C "$path" branch --show-current 2>/dev/null)"
  dirty="$(tracked_count "$path")"
  if [ "$dirty" -ne 0 ]; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=tracked_changes count=%s\n' "$repo" "$dirty"
    git -C "$path" status --short --untracked-files=no | sed -n '1,20p'
    return 1
  fi
  if [ -n "$branch" ] && [ "$branch" != "$target_branch" ]; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=unexpected_checkout_branch checkout=%s target=%s\n' \
      "$repo" "$branch" "$target_branch"
    return 1
  fi
  return 0
}

fetch_branch() {
  local repo="$1"
  local path="$2"
  local branch="$3"
  local status

  printf 'P3_UPDATE_FETCH repo=%s branch=%s\n' "$repo" "$branch"
  # Each repo has its own command and therefore its own credential prompt.
  # Do not paste a username/password into shell text; respond to Git directly.
  git -C "$path" fetch origin "$branch"
  status=$?
  if [ "$status" -ne 0 ]; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=fetch_failed status=%s\n' "$repo" "$status"
    return 1
  fi
  return 0
}

preflight_pinned_target() {
  local repo="$1"
  local path="$2"
  local branch="$3"
  local expected="$4"
  local head

  if ! git -C "$path" cat-file -e "${expected}^{commit}" 2>/dev/null; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=pinned_sha_unavailable sha=%s\n' "$repo" "$expected"
    return 1
  fi
  if ! git -C "$path" merge-base --is-ancestor "$expected" "origin/${branch}"; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=pinned_sha_not_on_remote_branch sha=%s branch=%s\n' \
      "$repo" "$expected" "$branch"
    return 1
  fi

  head="$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
  if [ "$head" != "$expected" ] && ! git -C "$path" merge-base --is-ancestor "$head" "$expected"; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=non_fast_forward_to_pinned_sha head=%s target=%s\n' \
      "$repo" "$head" "$expected"
    return 1
  fi
  printf 'P3_UPDATE_PREFLIGHT repo=%s target=%s remote_branch=origin/%s fast_forward=1\n' \
    "$repo" "$expected" "$branch"
  return 0
}

merge_pinned_target() {
  local repo="$1"
  local path="$2"
  local expected="$3"
  local status

  git -C "$path" merge --ff-only "$expected"
  status=$?
  if [ "$status" -ne 0 ]; then
    printf 'P3_UPDATE_BLOCKED repo=%s reason=ff_only_merge_failed status=%s target=%s\n' \
      "$repo" "$status" "$expected"
    return 1
  fi
  printf 'P3_UPDATE_MERGED repo=%s target=%s\n' "$repo" "$expected"
  return 0
}

main() {
  local overall=0
  local in_container=0
  local vllm_head
  local kvaf_head

  case "$mode" in
    update|check) ;;
    *)
      print_usage
      return 1
      ;;
  esac

  if [ -f /.dockerenv ]; then
    in_container=1
  fi
  printf 'P3_UPDATE_HOST hostname=%s in_container=%s mode=%s\n' \
    "$(hostname)" "$in_container" "$mode"

  if [ -z "$P3_VLLM_SHA" ] || [ -z "$P3_KVAF_SHA" ]; then
    printf 'P3_UPDATE_BLOCKED reason=missing_pinned_sha vllm=<%s> kvaf=<%s>\n' \
      "$P3_VLLM_SHA" "$P3_KVAF_SHA"
    print_usage
    return 1
  fi
  if ! detect_paths; then
    return 1
  fi

  print_state vllm "$VLLM_KIP" "$P3_VLLM_BRANCH" "$P3_VLLM_SHA"
  print_state kvaf "$KVAF_FUSED" "$P3_KVAF_BRANCH" "$P3_KVAF_SHA"

  precheck_checkout vllm "$VLLM_KIP" "$P3_VLLM_BRANCH" || overall=1
  precheck_checkout kvaf "$KVAF_FUSED" "$P3_KVAF_BRANCH" || overall=1

  if [ "$overall" -eq 0 ] && [ "$mode" = update ]; then
    # Fetch both first. No worktree is advanced until both source targets
    # have been proven available and fast-forwardable.
    fetch_branch vllm "$VLLM_KIP" "$P3_VLLM_BRANCH" || overall=1
    fetch_branch kvaf "$KVAF_FUSED" "$P3_KVAF_BRANCH" || overall=1
  fi

  if [ "$overall" -eq 0 ] && [ "$mode" = update ]; then
    preflight_pinned_target vllm "$VLLM_KIP" "$P3_VLLM_BRANCH" "$P3_VLLM_SHA" || overall=1
    preflight_pinned_target kvaf "$KVAF_FUSED" "$P3_KVAF_BRANCH" "$P3_KVAF_SHA" || overall=1
  fi

  if [ "$overall" -eq 0 ] && [ "$mode" = update ]; then
    merge_pinned_target vllm "$VLLM_KIP" "$P3_VLLM_SHA" || overall=1
    merge_pinned_target kvaf "$KVAF_FUSED" "$P3_KVAF_SHA" || overall=1
  fi

  vllm_head="$(git -C "$VLLM_KIP" rev-parse HEAD 2>/dev/null || true)"
  kvaf_head="$(git -C "$KVAF_FUSED" rev-parse HEAD 2>/dev/null || true)"
  printf 'P3_UPDATE_FINAL vllm=%s expected=%s kvaf=%s expected=%s ready=%s\n' \
    "$vllm_head" "$P3_VLLM_SHA" "$kvaf_head" "$P3_KVAF_SHA" \
    "$([ "$overall" -eq 0 ] && [ "$vllm_head" = "$P3_VLLM_SHA" ] && \
       [ "$kvaf_head" = "$P3_KVAF_SHA" ] && echo 1 || echo 0)"
  return "$overall"
}

main "$@"
