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
        self.slot_buffers = []
        self.slot_arrays = []
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

    def allocate_slot(self):
        """Second set of bound outputs so a launch can run while slot 0's results are read."""
        if self.slot_buffers:
            return
        from triton.backends.amd_triton_npu import shared as buffers

        try:
            for rows, columns, _ in self.shapes:
                self.slot_buffers.append(buffers.empty((rows, columns), dtype=torch.int32, device="xrt:0"))
            for buffer in self.slot_buffers:
                buffer.bo.sync(self.runner._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        except Exception:
            self.close_slot()
            raise
        self.slot_arrays = [buffer.numpy() for buffer in self.slot_buffers]

    def close_slot(self):
        self.slot_arrays = []
        for buffer in self.slot_buffers:
            buffer.close()
        self.slot_buffers = []

    def start(self, query, context, key, slot=0):
        """Launch without waiting; finish() returns that slot's outputs.

        Needs a key already populated by run(). One launch may be in flight; slot 1
        owns separate output buffers, so slot 0's results stay valid while slot 1 runs.
        """
        if key not in self.populated:
            raise ValueError(f"INT8 QKV weights are not populated for {key}")
        if slot not in (0, 1):
            raise ValueError("QKV output slot must be 0 or 1")
        if self.shared and query is not context:
            raise ValueError("Shared QKV requires the same quantized activation object")
        xrt = self.runner._xrt
        sizes, bos = self.runner._bos[key]
        if slot:
            self.allocate_slot()
            bos = list(bos)
            for mapping, buffer in zip(self.maps, self.slot_buffers):
                bos[mapping[2]] = buffer.bo
        activations = {}
        for i, ((rows, _, inner), mapping) in enumerate(zip(self.shapes, self.maps)):
            value = query if i == 0 else context
            if (not isinstance(value, np.ndarray) or value.dtype != np.int8
                    or value.shape != (rows, inner) or not value.flags.c_contiguous):
                raise ValueError(f"activation requires contiguous INT8 {(rows, inner)}")
            if sizes[mapping[0]] != value.nbytes:
                raise ValueError("INT8 QKV activation size changed for a populated key")
            activations[mapping[0]] = value
        for index, value in activations.items():
            target = np.frombuffer(bos[index].map(), dtype=np.uint8, count=value.nbytes)
            np.copyto(target, np.frombuffer(value, dtype=np.uint8), casting="no")
            bos[index].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = xrt.run(self.runner.kernel)
        for index, bo in enumerate(bos):
            run.set_arg(index, bo)
        run.start()
        return {
            "run": run, "bos": bos, "arrays": self.slot_arrays if slot else self.output_arrays,
            "h2d_bytes": sum(value.nbytes for value in activations.values()),
        }

    def finish(self, pending):
        xrt = self.runner._xrt
        pending["run"].wait2()
        for index in self.outputs:
            pending["bos"][index].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return tuple(torch.from_numpy(array) for array in pending["arrays"]), {
            "dispatches": 1,
            "h2d_bytes": pending["h2d_bytes"],
            "d2h_bytes": sum(array.nbytes for array in pending["arrays"]),
            "allocation_count": 0,
            "resident_hits": 3,
            "weight_population_bytes": 0,
        }

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
        self.close_slot()
        self.output_arrays = []
        self.bound = {}
        for buffer in self.buffers:
            buffer.close()
        self.buffers = []
