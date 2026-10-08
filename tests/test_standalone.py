import importlib.util
from pathlib import Path
import sys
import unittest

from comfyui_xdna_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


ROOT = Path(__file__).resolve().parents[1]


class StandaloneTests(unittest.TestCase):
    def test_comfyui_loads_root_as_custom_node_package(self):
        name = "anima-xdna-comfyui"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            self.assertEqual(
                {key: value.__name__ for key, value in module.NODE_CLASS_MAPPINGS.items()},
                {key: value.__name__ for key, value in NODE_CLASS_MAPPINGS.items()},
            )
            self.assertEqual(
                module.NODE_DISPLAY_NAME_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
            )
            self.assertEqual(
                set(module.NODE_CLASS_MAPPINGS),
                {
                    "LoadAnimaBF16",
                    "LoadAttachAnimaXDNAModel",
                    "AnimaXDNARuntimeStatus",
                    "AnimaXDNAUnload",
                },
            )
        finally:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]

    def test_no_bridge_implementation_is_distributed(self):
        for name in ("anima_style_bridge.py", "comfyui_nodes", "tests/test_style_bridge.py"):
            self.assertFalse((ROOT / name).exists(), name)
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "anima-xdna-comfyui"', metadata)
        self.assertNotIn("anima-style-bridge =", metadata)
        self.assertNotIn('py-modules = ["anima_style_bridge"]', metadata)


if __name__ == "__main__":
    unittest.main()
