# SPDX-License-Identifier: Apache-2.0
"""Request completion aggregation; SDK objects, not flags, prove DMA completion."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Contribution:
    """One layer/chunk from an actual P participant, within a request epoch."""

    layer: str
    peer: str
    chunk: int


@dataclass
class CompletionGate:
    """Separate D first-Decode readiness from P physical storage reuse.

    needs maps each required contribution to a set containing 'hbm' and/or
    'ddr'. Empty contributors must still acknowledge. MTP layers belong in the
    same required set; this class never assumes target-layer count is enough.
    Call hbm_done only after raw READ/recovery success, and ddr_committed only
    after the SDK Commit acknowledgment. Publication is a separate D gate.
    """

    request_id: str
    epoch: int
    needs: Mapping[Contribution, frozenset[str]]
    _done: dict[Contribution, set[str]] = field(init=False, repr=False)
    _failed: bool = field(default=False, init=False)

    def __post_init__(self):
        if not self.request_id or type(self.epoch) is not int or self.epoch < 0 or not self.needs:
            raise ValueError("request identity and nonempty contributor set are required")
        self.needs = {key: frozenset(paths) for key, paths in self.needs.items()}
        for key, paths in self.needs.items():
            if not paths <= {"hbm", "ddr"} or not key.layer or not key.peer or key.chunk < 0:
                raise ValueError("invalid contribution or transfer path")
        self._done = {key: set() for key in self.needs}

    def record(self, request_id: str, epoch: int, key: Contribution, event: str) -> None:
        if (request_id, epoch) != (self.request_id, self.epoch) or key not in self.needs:
            raise ValueError("stale request or unexpected contributor")
        paths = self.needs[key]
        allowed = {"ack"} | ({"hbm_done"} if "hbm" in paths else set())
        if "ddr" in paths:
            allowed |= {"ddr_committed", "ddr_published"}
        if event not in allowed:
            raise ValueError("unexpected completion event")
        if event == "ddr_published" and "ddr_committed" not in self._done[key]:
            raise ValueError("cannot publish before Commit")
        self._done[key].add(event)

    def source_reusable(self, key: Contribution) -> bool:
        """A storage owner must aggregate this across every reader of its MR."""
        paths, done = self.needs[key], self._done[key]
        required = {"ack"}
        if "hbm" in paths:
            required.add("hbm_done")
        if "ddr" in paths:
            required.add("ddr_committed")
        return required <= done

    @property
    def decode_ready(self) -> bool:
        return not self._failed and all(
            self.source_reusable(key) and ("ddr" not in paths or "ddr_published" in self._done[key])
            for key, paths in self.needs.items()
        )

    def fail(self) -> None:
        # Failure/timeout never manufactures a completion or a source release.
        self._failed = True
