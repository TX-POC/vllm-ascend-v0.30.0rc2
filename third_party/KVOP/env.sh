#!/usr/bin/env bash
# Source before Python initializes CANN; preserve all existing vendor paths.
_kvop_load_env() {
    local root library vendor
    local -a libraries
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    libraries=("$root"/original/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
    if [[ "${#libraries[@]}" != 1 || ! -f "${libraries[0]}" ]]; then
        echo "Build KVOP first: bash $root/build.sh" >&2
        return 1
    fi
    library="${libraries[0]}"
    vendor="${library%/op_api/lib/libcust_opapi.so}"
    case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
        *":$vendor:"*) ;;
        *) export ASCEND_CUSTOM_OPP_PATH="$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}" ;;
    esac
}
_kvop_load_env
_kvop_status=$?
unset -f _kvop_load_env
return "$_kvop_status"
