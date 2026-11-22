import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from comfyui_xdna_nodes import runtime_config


def write_config(root, **values):
    runtime = Path(root) / "xdna-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / "anima-xdna.json"
    path.write_text(json.dumps({"schema_version": 1, **values}), encoding="utf-8")
    return path


class RuntimeConfigTest(unittest.TestCase):
    def test_int8_config_sets_launcher_environment_without_overriding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_config(directory, profile="int8-attn-l1k64-nq8pp",
                                xrt_dev_dir="C:/sdk/xrt-2.21.75", triton_cache_dir="cache/triton")
            runtime = path.parent.resolve()
            environ = {"ANIMA_XDNA_NUM_THREADS": "6"}
            messages = []
            found = runtime_config.apply_runtime_config(environ, [Path(directory)], messages.append)

        self.assertEqual(found, path.resolve())
        self.assertEqual(environ["ANIMA_XDNA_NUM_THREADS"], "6")
        self.assertEqual(environ["ANIMA_XDNA_OPTIMIZED_PROFILE"], "int8-attn-l1k64-nq8pp")
        self.assertEqual(environ["ANIMA_XDNA_ATTENTION_BACKEND"], "air")
        self.assertEqual(environ["ANIMA_XDNA_ADALN_DEVICE"], "host")
        self.assertEqual(environ["ANIMA_XDNA_BF16_VAE_CLFIX"], "1")
        self.assertEqual(environ["ANIMA_XDNA_INT8_ARTIFACT_DIR"], str(runtime / "int8-attn-l1k64-nq8pp"))
        self.assertEqual(environ["ANIMA_XDNA_AIR_ATTENTION_DIR"], str(runtime / "air-attention-nq8pp"))
        self.assertEqual(environ["ANIMA_XDNA_GEMM_LAYOUT_DIR"],
                         str(runtime / "gemm-wide32-20261025" / "library"))
        self.assertEqual(environ["TRITON_CACHE_DIR"], str(runtime / "cache" / "triton"))
        self.assertEqual(Path(environ["XRT_DEV_DIR"]), Path("C:/sdk/xrt-2.21.75"))
        self.assertIn("ANIMA_XDNA_NUM_THREADS", messages[-1])

    def test_standard_config_leaves_optimized_profile_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            write_config(directory, profile="standard")
            environ = {}
            runtime_config.apply_runtime_config(environ, [Path(directory)], lambda _: None)
        self.assertFalse(any(name.startswith("ANIMA_XDNA_") for name in environ))

    def test_invalid_or_missing_configs_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            for values in ({"profile": "int8"}, {"profile": "standard", "typo": 1},
                           {"profile": "standard", "num_threads": True}):
                path = write_config(directory, **values)
                with self.assertRaises(ValueError):
                    runtime_config.load_config(path)
            missing = {"ANIMA_XDNA_CONFIG": str(Path(directory) / "missing.json")}
            with self.assertRaisesRegex(FileNotFoundError, "ANIMA_XDNA_CONFIG"):
                runtime_config.find_config(missing, [])

    def test_roots_come_from_comfyui_and_custom_node_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            node_file = Path(directory) / "Comfy" / "custom_nodes" / "anima-xdna-comfyui" / "comfyui_xdna_nodes" / "x.py"
            folder_paths = types.ModuleType("folder_paths")
            folder_paths.base_path = str(base)
            with mock.patch.dict("sys.modules", {"folder_paths": folder_paths}):
                roots = runtime_config.candidate_roots(node_file)
        self.assertEqual(roots, [base.resolve(), (Path(directory) / "Comfy").resolve()])

    @unittest.skipUnless(os.name == "nt", "XRT SDK discovery is Windows-only")
    def test_xrt_sdk_detection_prefers_newest_complete_local_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            amd = Path(directory) / "AMD"
            for name, complete in (("xrt-2.9.1", True), ("xrt-2.21.75", True), ("xrt-2.30.0", False)):
                (amd / name / "include" / "xrt").mkdir(parents=True)
                if complete:
                    (amd / name / "lib").mkdir()
            environ = {"LOCALAPPDATA": directory, "PROGRAMFILES": str(Path(directory) / "none")}
            self.assertEqual(runtime_config.detect_xrt_sdk(environ), str(amd / "xrt-2.21.75"))
            self.assertIsNone(runtime_config.detect_xrt_sdk({**environ, "XRT_DEV_DIR": "x"}))
            program_files = Path(directory) / "pf"
            (program_files / "AMD" / "xrt" / "include" / "xrt").mkdir(parents=True)
            (program_files / "AMD" / "xrt" / "lib").mkdir()
            self.assertIsNone(runtime_config.detect_xrt_sdk({**environ, "PROGRAMFILES": str(program_files)}))


if __name__ == "__main__":
    unittest.main()
