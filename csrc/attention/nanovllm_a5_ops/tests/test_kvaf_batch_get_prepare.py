#!/usr/bin/env python3
"""Ascend 950 validation for CacheOS subgraph-C BatchGet preparation."""

from __future__ import annotations

import argparse

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5  # noqa: F401


BLOCK_SIZE = 128
PACKED_ROW_BYTES = 656
TOPK = 2048
INDICES_POISON = -101
OFFSETS_POISON = -102
OUTPUT_INDICES_POISON = -103
MASK_POISON = 0xA5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.npu.set_device(device)

    batch = 4
    max_num_reqs = 8
    num_layers = 4
    layer_id = 2
    qcap = 256
    resident_blocks_per_row = 64

    sources_cpu = torch.zeros((batch, 1, TOPK), dtype=torch.int32)
    destinations_cpu = torch.zeros_like(sources_cpu)
    pool_rows_cpu = torch.tensor([1, 6, 3, 2], dtype=torch.int32)
    resident_table_cpu = torch.arange(
        batch * resident_blocks_per_row, dtype=torch.int32
    ).view(batch, resident_blocks_per_row)
    cache_tokens_cpu = torch.tensor([0, TOPK, 4096, TOPK], dtype=torch.int32)
    candidate_lens_cpu = torch.tensor(
        [73, 8192, 12032, 4096], dtype=torch.int32
    )
    logical_lens_cpu = torch.tensor(
        [73, 8201, 12049, 4100], dtype=torch.int32
    )

    active_sources = ([257, 1023, 4096], [128, 255, 16384, 32767, 511])
    active_destinations = ([0, 129, 511], [128, 255, 1024, 2047, 17])
    for row, (sources, destinations) in enumerate(
        zip(active_sources, active_destinations), start=1
    ):
        sources_cpu[row, 0, : len(sources)] = torch.tensor(sources)
        destinations_cpu[row, 0, : len(destinations)] = torch.tensor(destinations)
    sources_cpu[3, 0] = torch.arange(TOPK, dtype=torch.int32)
    destinations_cpu[3, 0] = torch.arange(TOPK, dtype=torch.int32)

    request_capacity = batch * TOPK
    source_npu = sources_cpu.to(device)
    destination_npu = destinations_cpu.to(device)
    pool_rows_npu = pool_rows_cpu.to(device)
    resident_table_npu = resident_table_cpu.to(device)
    cache_tokens_npu = cache_tokens_cpu.to(device)
    candidate_lens_npu = candidate_lens_cpu.to(device)
    logical_lens_npu = logical_lens_cpu.to(device)

    def run_case(case_name: str, miss_cpu: torch.Tensor) -> None:
        values = (
            source_npu,
            destination_npu,
            miss_cpu.to(device),
            pool_rows_npu,
            resident_table_npu,
            cache_tokens_npu,
            candidate_lens_npu,
            logical_lens_npu,
            torch.full(
                (request_capacity,),
                INDICES_POISON,
                dtype=torch.int32,
                device=device,
            ),
            torch.full(
                (request_capacity,),
                OFFSETS_POISON,
                dtype=torch.int32,
                device=device,
            ),
            torch.full(
                (request_capacity,),
                MASK_POISON,
                dtype=torch.uint8,
                device=device,
            ),
            torch.full(
                (request_capacity,),
                OUTPUT_INDICES_POISON,
                dtype=torch.int32,
                device=device,
            ),
            torch.empty(batch, dtype=torch.int32, device=device),
        )
        result = torch.ops.nanovllm_dsa.kvaf_batch_get_prepare_out.default(
            *values, layer_id, num_layers, max_num_reqs, qcap
        )
        torch.npu.synchronize()
        indices, offsets, mask, output_indices, resident_lengths = (
            tensor.cpu() for tensor in result
        )

        expected_mask = torch.zeros(request_capacity, dtype=torch.uint8)
        expected_indices = torch.full(
            (request_capacity,), INDICES_POISON, dtype=torch.int32
        )
        expected_offsets = torch.full(
            (request_capacity,), OFFSETS_POISON, dtype=torch.int32
        )
        expected_outputs = torch.full(
            (request_capacity,), OUTPUT_INDICES_POISON, dtype=torch.int32
        )
        for row, count in enumerate(miss_cpu.tolist()):
            start = row * TOPK
            expected_mask[start : start + count] = 1
            for column in range(count):
                token = int(sources_cpu[row, 0, column])
                destination = int(destinations_cpu[row, 0, column])
                expected_indices[start + column] = (
                    int(pool_rows_cpu[row]) * num_layers + layer_id
                ) * qcap + token // BLOCK_SIZE
                expected_offsets[start + column] = (
                    token % BLOCK_SIZE
                ) * PACKED_ROW_BYTES
                physical_block = int(
                    resident_table_cpu[row, destination // BLOCK_SIZE]
                )
                expected_outputs[start + column] = (
                    physical_block * BLOCK_SIZE + destination % BLOCK_SIZE
                )

        torch.testing.assert_close(mask, expected_mask)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(offsets, expected_offsets)
        torch.testing.assert_close(output_indices, expected_outputs)
        torch.testing.assert_close(
            resident_lengths,
            torch.tensor(
                [73, TOPK + 9, 4096 + 17, TOPK + 4], dtype=torch.int32
            ),
        )
        assert num_layers * max_num_reqs * qcap > int(indices.max())
        print(f"KVAF_BATCH_GET_PREPARE_{case_name.upper()}_PASS")

    run_case("simt", torch.tensor([0, 3, 5, TOPK], dtype=torch.int32))
    run_case("scalar", torch.tensor([0, 3, 5, 7], dtype=torch.int32))
    print("KVAF_BATCH_GET_PREPARE_PASS")


if __name__ == "__main__":
    main()
