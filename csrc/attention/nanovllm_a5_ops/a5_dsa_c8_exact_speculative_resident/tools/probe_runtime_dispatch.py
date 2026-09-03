#!/usr/bin/env python3
"""Host-safe visibility probe for the two C8 runtime dispatch namespaces.

This script intentionally imports modules and reads attributes only. It never
allocates a tensor, selects a device, performs a stream barrier, or invokes an
op.
"""

from __future__ import annotations


def _callable_attribute(owner: object | None, name: str) -> int:
    return int(owner is not None and callable(getattr(owner, name, None)))


def main() -> int:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as error:  # pragma: no cover - host-specific import
        print(
            "P10_RUNTIME_DISPATCH_PROBE status=unavailable "
            f"reason=torch_import_failed type={type(error).__name__}"
        )
        return 0

    try:
        import torch_npu  # type: ignore[import-not-found]
    except Exception:
        torch_npu = None

    ascend = getattr(torch.ops, "_C_ascend", None)
    registered = {
        "torch_npu.npu_quant_lightning_indexer": _callable_attribute(
            torch_npu, "npu_quant_lightning_indexer"
        ),
        "_C_ascend.npu_lightning_indexer_quant": _callable_attribute(
            ascend, "npu_lightning_indexer_quant"
        ),
        "_C_ascend.npu_vllm_quant_lightning_indexer": _callable_attribute(
            ascend, "npu_vllm_quant_lightning_indexer"
        ),
    }
    fields = " ".join(
        f"{name}={value}" for name, value in registered.items()
    )
    print(
        "P10_RUNTIME_DISPATCH_PROBE status=ok "
        f"torch_npu_imported={int(torch_npu is not None)} {fields}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
