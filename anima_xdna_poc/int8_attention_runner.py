"""Session-only native INT8 QKV chain with resident weights and shared outputs."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def qkv_shapes(shared):
    return ((1024, 2048, 2048),) * 3 if shared else (
        (1024, 2048, 2048), (512, 2048, 1024), (512, 2048, 1024),
    )


def qkv_maps(shared):
    return ((0, 1, 2), (0, 3, 4), (0, 5, 6)) if shared else (
        (0, 1, 2), (3, 4, 5), (3, 6, 7),
    )


class Int8QKVRunner:
    def __init__(self, directory, shared):
        from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner
        from triton.backends.amd_triton_npu import shared as buffers

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.shapes = qkv_shapes(shared)
        if (manifest["status"] != "compiled" or manifest["dtype"] != "i8"
                or manifest["shared_input"] != shared
                or tuple(tuple(s) for s in manifest["shapes"]) != self.shapes):
            raise ValueError(f"Incompatible INT8 QKV artifact: {directory}")
        if hashlib.sha256((directory / "aie.elf").read_bytes()).hexdigest() != manifest["elf_sha256"]:
            raise ValueError(f"INT8 QKV artifact hash mismatch: {directory}")
        self.shared = shared
        self.maps = qkv_maps(shared)
        self.static = {m[1] for m in self.maps}
        self.outputs = {m[2] for m in self.maps}
        self.populated = set()
        self.runner = MultiLaunchRunner(str(directory / "aie.elf"), manifest["elf_kernel_name"])
        self.buffers = []
        try:
            for rows, columns, _ in self.shapes:
                self.buffers.append(buffers.empty((rows, columns), dtype=torch.int32, device="xrt:0"))
        except Exception:
            self.close()
            raise
        self.templates = [np.empty((inner, columns), dtype=np.int8) for _, columns, inner in self.shapes]
        self.output_arrays = [buffer.numpy() for buffer in self.buffers]
        self.bound = {m[2]: buffer.bo for m, buffer in zip(self.maps, self.buffers)}

    def run(self, query, context, weights, key):
        """Outputs borrow shared BO storage until the next run of this runner."""
        first = key not in self.populated
        if first and weights is None:
            raise ValueError(f"INT8 QKV weights are not populated for {key}")
        if self.shared and query is not context:
            raise ValueError("Shared QKV requires the same quantized activation object")
        if weights is not None and len(weights) != 3:
            raise ValueError("QKV requires three weight tensors")
        arrays = [None] * (7 if self.shared else 8)
        for i, ((rows, columns, inner), mapping) in enumerate(zip(self.shapes, self.maps)):
            activation = query if i == 0 else context
            weight = weights[i] if first else self.templates[i]
            for name, value, shape in (
                ("activation", activation, (rows, inner)), ("weight", weight, (inner, columns)),
            ):
                if not isinstance(value, np.ndarray) or value.dtype != np.int8 or value.shape != shape:
                    raise ValueError(f"{name} requires INT8 {shape}")
                if not value.flags.c_contiguous:
                    raise ValueError(f"{name} must be C-contiguous")
            arrays[mapping[0]] = activation
            arrays[mapping[1]] = weight
            arrays[mapping[2]] = self.output_arrays[i]
        started = time.perf_counter()
        result = self.runner.run(
            arrays, bo_key=key, static_indices=self.static, intermediate_indices=self.outputs,
            output_indices=self.outputs, bound_buffers=self.bound,
        )
        elapsed = (time.perf_counter() - started) * 1000
        self.populated.add(key)
        dynamic = {m[0] for m in self.maps}
        weight_bytes = sum(arrays[i].nbytes for i in self.static) if first else 0
        return tuple(torch.from_numpy(result[m[2]]) for m in self.maps), {
            "dispatches": 1,
            "h2d_bytes": sum(arrays[i].nbytes for i in dynamic) + weight_bytes,
            "d2h_bytes": sum(arrays[i].nbytes for i in self.outputs),
            "allocation_count": len(self.static | dynamic) if first else 0,
            "resident_hits": 0 if first else 3,
            "weight_population_bytes": weight_bytes,
            "runner_run_ms": elapsed,
        }

    def close(self):
        self.runner.unload()
        # Drop ndarray views before releasing the storage they alias.
        self.output_arrays = []
        self.bound = {}
        for buffer in self.buffers:
            buffer.close()
        self.buffers = []
