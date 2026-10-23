"""Opt-in, fixed-shape Anima attention authored directly with MLIR-AIR.

The device ABI has only Q, K, V and the final output. Scores, unnormalised
probabilities and online-softmax state stay in core memory. This prototype
uses pinned MIT-licensed upstream AIE microkernels, not Triton lowering.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from anima_xdna_poc.air_attention import AIRAttention, AttentionShape

UPSTREAM_COMMIT = "508ca731ef0384fd3729ac1688e0b5d9a0ef5efe"
SOURCE_URL = (
    "https://raw.githubusercontent.com/Xilinx/mlir-air/"
    + UPSTREAM_COMMIT
    + "/programming_examples/flash_attention/kernel_fusion_based/"
)
SOURCE_SHA256 = {
    "attn_npu2.cc": "56397ccfb36f2a9326ecbd6d6face26610ce161e488327698abff03e097f12e7",
    "zero.cc": "23cb847ed6bc1f3db6f0c8e05b1aa91a029e464d083ee1f09d2e4cf1f9110eba",
}


def build_module(shape, kernel_object):
    from air import api as air
    from air.api.types import bf16, i32

    h = shape.parallel_heads
    nq = shape.query_cores
    tq, tk, d = shape.query_tile, shape.key_tile, shape.depth
    qgroup = shape.query_group
    blocks = shape.keys // tk
    link = str(Path(kernel_object).resolve())
    extern = lambda name: air.extern(name, link_with=link)
    zero_g = extern("zero_fill_g_bf16")
    zero_o = extern("zero_fill_gp_bf16")
    zero_l = extern("zero_fill_sp_bf16")
    init_m = extern("neg_inf_fill_up_bf16")
    qk = extern("matmul_a_b_bf16")
    pv = extern("matmul_g_b_bf16")
    softmax = extern("fused_softmax")
    rescale_o = extern("mul_r_gp")
    update_l = extern("accum_sp_r_s")
    copy_l = air.extern("vector_copy_32elems", link_with=link, scalars=[i32])
    copy_q = air.extern("copy_half_tile", link_with=link, scalars=[i32])
    normalise = extern("div_gp_sp")
    qin = air.channel("QIn", size=[h])
    q2l1 = air.channel("Q2L1", size=[h, nq // 2, 1], broadcast_shape=[h, nq // 2, 2])
    kin = air.channel("KIn", size=[h])
    vin = air.channel("VIn", size=[h])
    kv2l1 = air.channel("KV2L1", size=[h, 1, 1], broadcast_shape=[h, nq // 2, 2])
    o2l2 = air.channel("O2L2", size=[h, nq // 2, 2])
    out = air.channel("Output", size=[h])
    q = air.tensor([shape.heads, shape.queries, d], bf16)
    k = air.tensor([shape.heads, shape.keys, d], bf16)
    v = air.tensor([shape.heads, shape.keys, d], bf16)
    o = air.tensor([shape.heads, shape.queries, d], bf16)
    k_flat = k.reshape(shape.heads * shape.keys * d)
    v_flat = v.reshape(shape.heads * shape.keys * d)
    with air.launch(
        [range(shape.queries // qgroup), range(shape.heads // h)],
        name="anima_attention",
    ) as launch:

        @launch.body
        def _(query_group, head_group):
            first_row = query_group * qgroup
            for local_head in range(h):
                head = head_group * h + local_head
                qin.put(
                    q[head, first_row:first_row + qgroup, :],
                    indices=[local_head],
                )
                kin.put(
                    k_flat[head * shape.keys * d:(head + 1) * shape.keys * d]
                    .reshape(blocks, tk, d),
                    indices=[local_head],
                )
                vin.put(
                    v_flat[head * shape.keys * d:(head + 1) * shape.keys * d]
                    .reshape(blocks, tk, d),
                    indices=[local_head],
                )
            with air.segment([range(h), range(1)], name="attention_segment") as seg:

                @seg.body
                def _(head_local, _unused):
                    q_l2 = air.alloc([qgroup, d], bf16, scope=seg.private())
                    k_l2 = air.alloc([tk, d], bf16, scope=seg.private())
                    v_l2 = air.alloc([tk, d], bf16, scope=seg.private())
                    o_l2 = air.alloc([qgroup, d], bf16, scope=seg.private())
                    qin.get(q_l2, indices=[head_local])
                    for pair in air.parallel(0, nq // 2):
                        q2l1.put(
                            q_l2.reshape(qgroup * d)[pair * 2 * tq * d:(pair + 1) * 2 * tq * d]
                            .reshape(2 * tq // 8, 8, d // 8, 8)
                            .transpose(2, 0, 1, 3),
                            indices=[head_local, pair, 0],
                        )
                    for _block in air.sequential(0, blocks):
                        kin.get(k_l2, indices=[head_local])
                        kv2l1.put(
                            k_l2.reshape(tk // 8, 8, d // 8, 8)
                            .transpose(2, 0, 1, 3),
                            indices=[head_local, 0, 0],
                        )
                        vin.get(v_l2, indices=[head_local])
                        kv2l1.put(
                            v_l2.reshape(tk // 8, 8, d // 8, 8)
                            .transpose(2, 0, 1, 3),
                            indices=[head_local, 0, 0],
                        )
                    with air.herd(
                        [range(2), range(nq // 2)], name="attention_core",
                        shape=(2, nq // 2), link_with=link,
                    ) as herd:

                        @herd.body
                        def _(column, pair):
                            q_pair = air.alloc([2 * tq, d], bf16, scope=herd.private())
                            q_l1 = air.alloc([tq, d], bf16, scope=herd.private())
                            kv_l1 = air.alloc([tk, d], bf16, scope=herd.private())
                            scores = air.alloc([tq, tk], bf16, scope=herd.private())
                            numerator = air.alloc([tq, d], bf16, scope=herd.private())
                            maximum = air.alloc([tq, 1], bf16, scope=herd.private())
                            denominator = air.alloc([tq, 1], bf16, scope=herd.private())
                            new_sum = air.alloc([tq, 1], bf16, scope=herd.private())
                            correction = air.alloc([tq, 1], bf16, scope=herd.private())
                            q2l1.get(q_pair, indices=[head_local, pair, column])
                            copy_q(q_pair, q_l1, column)
                            zero_o(numerator)
                            zero_l(denominator)
                            init_m(maximum)
                            for _block in air.sequential(0, blocks):
                                zero_g(scores.reshape(tq * tk))
                                kv2l1.get(kv_l1, indices=[head_local, pair, column])
                                qk(q_l1, kv_l1, scores.reshape(tq * tk))
                                # Scale is inside the upstream exp2 softmax.
                                softmax(scores.reshape(tq * tk), maximum, new_sum, correction)
                                rescale_o(correction, numerator)
                                kv2l1.get(kv_l1, indices=[head_local, pair, column])
                                pv(scores.reshape(tq * tk), kv_l1, numerator)
                                update_l(denominator, correction, new_sum)
                                copy_l(0, new_sum, denominator)
                            normalise(denominator, numerator)
                            o2l2.put(
                                numerator.reshape(d // 8, tq // 8, 8, 8)
                                .transpose(1, 2, 0, 3),
                                indices=[head_local, pair, column],
                            )
                    for row in air.parallel(0, nq):
                        o2l2.get(
                            o_l2[row * tq:row * tq + tq, :],
                            indices=[head_local, row // 2, row % 2],
                        )
                    out.put(o_l2, indices=[head_local])
            for local_head in range(h):
                head = head_group * h + local_head
                out.get(
                    o[head, first_row:first_row + qgroup, :],
                    indices=[local_head],
                )
    return launch.build(target="npu2")


def compile_attention(shape, work_dir, source_dir):
    from triton.backends.amd_triton_npu.driver import _aircc_compile, find_peano_root

    started = time.perf_counter()
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sources = Path(source_dir).resolve()
    for name in ("attn_npu2.cc", "zero.cc"):
        if not (sources / name).is_file():
            raise FileNotFoundError(sources / name)
        if hashlib.sha256((sources / name).read_bytes()).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError(f"source digest mismatch: {name}; expected {UPSTREAM_COMMIT}")
    import mlir_aie
    include = Path(mlir_aie.__path__[0]) / "include"
    peano = Path(find_peano_root())
    obj = work / "attention.o"
    command = [
        str(peano / "bin" / ("clang++.exe" if os.name == "nt" else "clang++")),
        "-O2", "-std=c++20", "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses", "-Wno-attributes", "-Wno-macro-redefined",
        "-Wno-empty-body", "-DNDEBUG", "-I", str(include),
        "-D__AIE_API_AIE_ADF_HPP__", "-DBIT_WIDTH=8",
        f"-Dlqp={shape.query_tile}", f"-Dlkp={shape.key_tile}",
        f"-Ddk={shape.depth}", f"-Ddk_full={shape.depth}",
        f"-Ddv={shape.depth}", f"-Ddv_full={shape.depth}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16", "-DROUND_CONV_EVEN",
        "-c", str(sources / "attn_npu2.cc"), "-o", str(obj),
    ]
    subprocess.run(command, cwd=work, check=True)
    module = build_module(shape, obj)
    path = work / "attention.mlir"
    path.write_text(str(module), encoding="utf-8")
    previous = Path.cwd()
    try:
        os.chdir(work)
        artifacts = _aircc_compile(
            str(path), "elf", "npu2", str(work), bf16_emulation=False,
            stack_size=4096, runtime_loop_tiling_sizes=(1, 1),
            extra_args=["--omit-ping-pong-transform=all"],
        )
    finally:
        os.chdir(previous)
    manifest = {
        "shape": shape.__dict__, "artifacts": artifacts,
        "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
        "upstream_commit": UPSTREAM_COMMIT,
        "compile_wall_ms": (time.perf_counter() - started) * 1000,
        "elf_sha256": hashlib.sha256(Path(artifacts["elf_path"]).read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((sources / name).read_bytes()).hexdigest()
            for name in ("attn_npu2.cc", "zero.cc")
        },
        "precision": "BF16 online-softmax state and block outputs; BFP16-emulated BF16 mmul, conv_even. Not control-equivalent.",
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--parallel-heads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    compile_attention(
        AttentionShape(args.keys, parallel_heads=args.parallel_heads),
        args.work_dir, args.source_dir,
    )


if __name__ == "__main__":
    main()
