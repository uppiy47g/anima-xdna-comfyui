import gc
import importlib
import json
import sys
import types
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from anima_xdna_poc.checkpoint_schema import NATIVE_NAMES, canonical_keys
from comfyui_xdna_nodes.model_wrapper import (
    ATTACHMENT_KEY,
    AUTO_CHECKPOINT,
    LoadAnimaBF16,
    LoadAnimaXDNACache,
    AnimaXDNARuntimeStatus,
    LoadAttachAnimaXDNAModel,
    ModelSourceProvenance,
    RuntimeAttachment,
    SharedRuntime,
    SOURCE_PROVENANCE_KEY,
    WRAPPER_KEY,
    _file_identity_token,
    _effective_lora_provider,
    _model_storage_profile,
    _validate_patcher,
)


Anima = type("Anima", (), {"__module__": "comfy.ldm.anima.model"})


class FakeBaseModel:
    def __init__(self):
        self.diffusion_model = Anima()


def fake_anima_parameters(dtype):
    parameters = {}
    for key in canonical_keys():
        _, block, name = key.split(".", 2)
        parameter_name = f"blocks.{block}.{NATIVE_NAMES[name]}"
        parameters[parameter_name] = torch.nn.Parameter(
            torch.zeros(1, dtype=dtype)
        )
    parameters["patch_embedding.weight"] = torch.nn.Parameter(
        torch.zeros(1, dtype=dtype)
    )
    return parameters


FakeAnimaParameters = type(
    "Anima",
    (Anima,),
    {
        "__init__": lambda self, parameters: setattr(self, "parameters", parameters),
        "named_parameters": lambda self: iter(self.parameters.items()),
    },
)


class FakePatcher:
    def __init__(self, model=None, load_device=None, offload_device=None):
        self.model = model if model is not None else FakeBaseModel()
        self.load_device = load_device or torch.device("cpu")
        self.offload_device = offload_device or torch.device("cpu")
        self.size = 0
        self.patches = {}
        self.patches_uuid = None
        self.hook_patches = {}
        self.weight_wrapper_patches = {}
        self.injections = {}
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}
        self.attachments = {}

    def clone(self):
        clone = type(self)(load_device=self.load_device, offload_device=self.offload_device)
        clone.model = self.model
        clone.patches = {key: value[:] for key, value in self.patches.items()}
        clone.patches_uuid = self.patches_uuid
        clone.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        clone.wrappers = {
            kind: {key: values.copy() for key, values in keys.items()}
            for kind, keys in self.wrappers.items()
        }
        clone.attachments = {
            key: value.on_model_patcher_clone()
            if hasattr(value, "on_model_patcher_clone")
            else value
            for key, value in self.attachments.items()
        }
        return clone

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        # Mirrors ComfyUI: only keys present in the model state_dict are patched.
        model_state = self.model.state_dict()
        added = []
        for key, adapter in patches.items():
            if key in model_state:
                self.patches.setdefault(key, []).append(
                    (strength_patch, adapter, strength_model, None, None)
                )
                added.append(key)
        self.patches_uuid = uuid.uuid4()
        return added

    def pre_run(self):
        pass

    def cleanup(self):
        pass

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def get_attachment(self, key):
        return self.attachments.get(key)

    def get_key_patches(self, filter_prefix=None):
        result = {}
        diffusion_model = self.model.diffusion_model
        if not hasattr(diffusion_model, "named_parameters"):
            return result
        for name, parameter in diffusion_model.named_parameters():
            key = f"diffusion_model.{name}"
            if filter_prefix is not None and not key.startswith(filter_prefix):
                continue
            result[key] = [
                (parameter, lambda tensor, **_kwargs: tensor),
                *self.patches.get(key, []),
            ]
        return result


class ComfyUIXDNAWrapperTests(unittest.TestCase):
    def setUp(self):
        patcher_extension = types.ModuleType("comfy.patcher_extension")

        class WrappersMP:
            DIFFUSION_MODEL = "diffusion_model"

        patcher_extension.WrappersMP = WrappersMP
        comfy = types.ModuleType("comfy")
        comfy.patcher_extension = patcher_extension
        comfy_float = types.ModuleType("comfy.float")
        comfy_float.stochastic_rounding = (
            lambda tensor, dtype, seed: tensor.to(dtype)
        )
        comfy_lora = types.ModuleType("comfy.lora")

        def calculate_weight(patches, weight, key):
            for strength, adapter, strength_model, offset, function in patches:
                if isinstance(adapter, tuple) and adapter[0] == "diff":
                    weight = weight + strength * adapter[1][0].to(weight.dtype)
                else:
                    weight = adapter.calculate_weight(
                        weight,
                        key,
                        strength,
                        strength_model,
                        offset,
                        function,
                    )
            return weight

        comfy_lora.calculate_weight = calculate_weight
        comfy_management = types.ModuleType("comfy.model_management")
        comfy_management.lora_compute_dtype = lambda _device: torch.float32
        comfy_management.load_models_gpu = mock.Mock()
        comfy_management.cast_to_device = (
            lambda tensor, device, dtype, copy=False: tensor.to(
                device=device, dtype=dtype, copy=copy
            )
        )
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.string_to_seed = lambda value: hash(value)
        weight_adapter = types.ModuleType("comfy.weight_adapter")
        weight_adapter_lora = types.ModuleType("comfy.weight_adapter.lora")

        class LoRAAdapter:
            def __init__(self, loaded_keys, weights):
                self.loaded_keys = loaded_keys
                self.weights = weights

            def calculate_weight(
                self,
                weight,
                _key,
                strength,
                _strength_model,
                _offset,
                function,
            ):
                up, down, alpha, _mid, _dora, _reshape = self.weights
                scale = 1.0 if alpha is None else alpha / down.shape[0]
                transform = function or (lambda value: value)
                return weight + transform(
                    strength * scale * torch.mm(up.float(), down.float())
                ).to(weight.dtype)

        weight_adapter_lora.LoRAAdapter = LoRAAdapter
        comfy.float = comfy_float
        comfy.lora = comfy_lora
        comfy.model_management = comfy_management
        comfy.utils = comfy_utils
        comfy.weight_adapter = weight_adapter
        weight_adapter.lora = weight_adapter_lora
        self.LoRAAdapter = LoRAAdapter
        extra_modules = {
            "comfy.float": comfy_float,
            "comfy.lora": comfy_lora,
            "comfy.model_management": comfy_management,
            "comfy.utils": comfy_utils,
            "comfy.weight_adapter": weight_adapter,
            "comfy.weight_adapter.lora": weight_adapter_lora,
        }
        self.saved = {
            name: sys.modules.get(name)
            for name in (
                "comfy",
                "comfy.patcher_extension",
                *extra_modules,
            )
        }
        sys.modules["comfy"] = comfy
        sys.modules["comfy.patcher_extension"] = patcher_extension
        sys.modules.update(extra_modules)

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    @contextmanager
    def cache_loader_module(self):
        comfy = sys.modules["comfy"]
        detection = types.ModuleType("comfy.model_detection")
        patcher = types.ModuleType("comfy.model_patcher")
        patcher.ModelPatcher = FakePatcher
        folder_paths = types.ModuleType("folder_paths")
        name = "comfyui_xdna_nodes.cache_loader"
        previous = sys.modules.pop(name, None)
        try:
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "comfy.model_detection": detection,
                        "comfy.model_patcher": patcher,
                        "folder_paths": folder_paths,
                    },
                ),
                mock.patch.object(comfy, "model_detection", detection, create=True),
                mock.patch.object(
                    comfy.model_management, "get_torch_device",
                    return_value=torch.device("cpu"), create=True,
                ),
                mock.patch.object(
                    comfy.model_management, "unet_manual_cast",
                    return_value=torch.float32, create=True,
                ),
            ):
                yield importlib.import_module(name)
        finally:
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous

    def test_cache_loader_never_reads_or_constructs_comfy_block_weights(self):
        class HostAnima(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList()
                self.host = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
                self.register_buffer("frequencies", torch.tensor([3.0]), persistent=False)

        HostAnima.__name__ = "Anima"
        with TemporaryDirectory() as temporary, self.cache_loader_module() as loader:
            config = types.SimpleNamespace(
                unet_config={
                    "image_model": "anima", "num_blocks": 28, "model_channels": 2048,
                    "in_channels": 16, "out_channels": 16, "num_heads": 16,
                },
                quant_config=None,
                supported_inference_dtypes=[torch.bfloat16, torch.float32],
                set_inference_dtype=mock.Mock(),
            )

            def construct(*args, **kwargs):
                self.assertEqual(config.unet_config["num_blocks"], 0)
                return types.SimpleNamespace(diffusion_model=HostAnima())

            config.get_model = construct
            loader.comfy.model_detection.model_config_from_unet = mock.Mock(return_value=config)
            path = Path(temporary, "model.safetensors")
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(dtype=dtype):
                    tensors = {
                        f"net.{key}": value.detach().to(dtype)
                        for key, value in fake_anima_parameters(dtype).items()
                        if key.startswith("blocks.")
                    }
                    tensors["net.host.weight"] = torch.tensor([[1.25]], dtype=dtype)
                    save_file(tensors, path)
                    reads = []

                    @contextmanager
                    def guarded_open(*args, **kwargs):
                        with safe_open(*args, **kwargs) as source:
                            proxy = mock.Mock(wraps=source)

                            def get_tensor(key):
                                self.assertNotIn(".blocks.", key)
                                reads.append(key)
                                return source.get_tensor(key)

                            proxy.get_tensor.side_effect = get_tensor
                            yield proxy

                    with mock.patch.object(loader, "safe_open", guarded_open):
                        model = loader._load_host_model(path)
                    self.assertEqual(reads, ["net.host.weight"])
                    self.assertEqual(list(model.diffusion_model.named_parameters())[0][0], "host.weight")
                    self.assertEqual(model.diffusion_model.host.weight.item(), 1.25)
                    self.assertEqual(model.diffusion_model.host.weight.dtype, torch.bfloat16)
                    self.assertEqual(model.diffusion_model.frequencies.item(), 3.0)
                    config.set_inference_dtype.assert_called_with(
                        torch.bfloat16, torch.float32, device=torch.device("cpu")
                    )
                    self.assertEqual(len(model.diffusion_model.blocks), 28)
                    block_state = {
                        key: value
                        for key, value in model.diffusion_model.state_dict().items()
                        if key.startswith("blocks.")
                    }
                    self.assertEqual(
                        set(block_state),
                        {key for key in fake_anima_parameters(dtype) if key.startswith("blocks.")},
                    )
                    self.assertTrue(all(
                        value.is_meta and value.dtype == torch.bfloat16
                        for value in block_state.values()
                    ))
                    self.assertEqual(
                        [name for name, _ in model.diffusion_model.named_parameters()],
                        ["host.weight"],
                    )
                    with self.assertRaisesRegex(RuntimeError, "CPU fallback is unavailable"):
                        model.diffusion_model.blocks[0](torch.zeros(1))
                    meta_sd = loader.comfy.model_detection.model_config_from_unet.call_args.args[0]
                    self.assertTrue(all(value.is_meta for value in meta_sd.values()))
                    tensors["net.host.weight"] = torch.zeros(2, 1, dtype=dtype)
                    save_file(tensors, path)
                    with self.assertRaisesRegex(RuntimeError, "host-layer shape mismatch"):
                        loader._load_host_model(path)
                    tensors["net.host.weight"] = torch.zeros(1, 1, dtype=dtype)
                    tensors["net.unexpected.weight"] = torch.zeros(1, dtype=dtype)
                    save_file(tensors, path)
                    with self.assertRaisesRegex(RuntimeError, "host-layer checkpoint mismatch"):
                        loader._load_host_model(path)
                    del tensors["net.host.weight"]
                    del tensors["net.unexpected.weight"]
                    save_file(tensors, path)
                    with self.assertRaisesRegex(RuntimeError, "host-layer checkpoint mismatch"):
                        loader._load_host_model(path)

    def test_cache_backed_runtime_verifies_cache_and_reports_only_host_storage(self):
        runtime = SharedRuntime(Path("fixture.safetensors"))
        diffusion = FakeAnimaParameters({
            "host.weight": torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)),
        })
        chain = mock.Mock()
        chain.source_identity = {"schema": "native-net", "block_fingerprint": "raw"}
        chain.execution_identity = {
            "source_dtypes": ["BF16"], "block_fingerprint": "verified",
        }
        chain.base_execution_identity = chain.execution_identity
        chain.cache_status = None
        with (
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.AnimaXDNAChainRuntime", return_value=chain,
            ),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.fingerprint_model_blocks",
                side_effect=AssertionError("No second block MODEL should be fingerprinted"),
            ),
        ):
            runtime.prepare(diffusion, cache_backed=True)
        chain.prepare_weight_cache.assert_called_once()
        status = runtime.snapshot()
        self.assertEqual(status["model_fingerprint"], "verified")
        self.assertEqual(status["model_block_parameter_bytes"], 0)
        self.assertEqual(status["model_nonblock_parameter_bytes"], 4)
        self.assertTrue(status["model_block_parameters_omitted"])
        self.assertFalse(status["model_block_parameters_released"])
        runtime.close()
        chain.prepare_weight_cache.side_effect = RuntimeError("cache integrity failure")
        with (
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.AnimaXDNAChainRuntime", return_value=chain,
            ),
            self.assertRaisesRegex(RuntimeError, "cache integrity failure"),
        ):
            runtime.prepare(diffusion, cache_backed=True)
        self.assertEqual(chain.close.call_count, 2)

    def test_cache_loader_clones_keep_guards_and_shared_runtime(self):
        with self.cache_loader_module() as loader:
            runtime = SharedRuntime(Path("fixture.safetensors"))
            model = loader.CacheBackedModelPatcher()
            model.set_attachments(ATTACHMENT_KEY, loader.CacheRuntimeAttachment(runtime))
            clone = model.clone()
            self.assertIs(type(clone), type(model))
            self.assertIs(clone.model, model.model)
            self.assertEqual(runtime.snapshot()["reference_count"], 2)
            clone.pre_run()
            self.assertIs(model.model.diffusion_model.xdna_active_runtime, runtime)
            clone.cleanup()
            self.assertIsNone(model.model.diffusion_model.xdna_active_runtime)
            for action, message in (
                (lambda: clone.get_key_patches("diffusion_model."), "no ComfyUI block weights"),
                (lambda: clone.add_hook_patches(None, {}), "weight hooks"),
                (lambda: clone.model_state_dict_for_saving(), "execution-only MODEL"),
            ):
                with self.assertRaisesRegex(RuntimeError, message):
                    action()
            model.get_attachment(ATTACHMENT_KEY).cleanup()
            with self.assertRaisesRegex(RuntimeError, "was unloaded"):
                model.pre_run()
            clone.pre_run()
            clone.get_attachment(ATTACHMENT_KEY).cleanup()
            self.assertEqual(runtime.snapshot()["state"], "closed")

    def _cache_backed_lora_fixture(self, loader, directory):
        dtype = torch.bfloat16
        tensors = {
            f"net.{key}": value.detach()
            for key, value in fake_anima_parameters(dtype).items()
            if key.startswith("blocks.")
        }
        tensors["net.blocks.0.self_attn.q_proj.weight"] = torch.tensor([[1.0]], dtype=dtype)
        tensors["net.host.weight"] = torch.zeros(1, dtype=dtype)
        path = Path(directory, "model.safetensors")
        save_file(tensors, path)
        placeholders = [{} for _ in range(28)]
        for key, value in tensors.items():
            if key.startswith("net.blocks."):
                _, _, index, name = key.split(".", 3)
                placeholders[int(index)][name] = torch.empty(
                    value.shape, dtype=dtype, device="meta"
                )

        class HostAnima(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.host = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
                self.blocks = torch.nn.ModuleList(
                    loader.CacheOnlyBlock(names) for names in placeholders
                )

        class HostModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = HostAnima()

        base_model = loader.CacheBackedModelPatcher(HostModel())
        base_model.set_attachments(
            SOURCE_PROVENANCE_KEY,
            ModelSourceProvenance("fixture", str(path), _file_identity_token(path)),
        )
        runtime = SharedRuntime(path)
        runtime.diagnostics.source_execution_fingerprint = "base-execution"
        base_model.set_attachments(ATTACHMENT_KEY, loader.CacheRuntimeAttachment(runtime))
        return path, base_model, runtime

    def test_cache_backed_model_accepts_standard_lora_and_builds_runtime_lazily(self):
        key = "diffusion_model.blocks.0.self_attn.q_proj.weight"
        adapter = self.LoRAAdapter(
            set(), (torch.tensor([[2.0]]), torch.tensor([[3.0]]), None, None, None, None)
        )
        built = []

        def prepare(runtime, diffusion_model, *, cache_backed=False):
            self.assertTrue(cache_backed)
            provider = runtime.effective_tensor_provider
            built.append((
                runtime,
                None if provider is None else provider("transformer_blocks.0.attn1.to_q.weight"),
                None if provider is None else provider("transformer_blocks.0.attn1.to_k.weight"),
            ))
            runtime.effective_tensor_provider = None
            runtime._runtime = mock.Mock(is_open=False)
            runtime.diagnostics.lora_patch_count = runtime.lora_patch_count
            runtime.diagnostics.source_execution_fingerprint = "base-execution"

        with TemporaryDirectory() as directory, self.cache_loader_module() as loader:
            path, base_model, base_runtime = self._cache_backed_lora_fixture(loader, directory)
            self.assertEqual(base_model.model_size(), 2)
            self.assertIn(key, base_model.model.state_dict())
            block = base_model.model.diffusion_model.blocks[0]
            # Bypass LoRA skips layers whose lookup raises AttributeError.
            with self.assertRaisesRegex(RuntimeError, "bypass LoRA"):
                getattr(block, "self_attn")
            self.assertFalse(hasattr(block, "unrelated"))
            lora_model = base_model.clone()
            self.assertEqual(lora_model.add_patches({key: adapter}, 0.5), [key])
            diffusion = base_model.model.diffusion_model
            with (
                mock.patch.object(SharedRuntime, "prepare", autospec=True, side_effect=prepare),
                mock.patch.object(SharedRuntime, "suspend", autospec=True) as suspend,
            ):
                lora_model.pre_run()
                lora_runtime = lora_model.get_attachment(ATTACHMENT_KEY).runtime
                self.assertIsNot(lora_runtime, base_runtime)
                self.assertEqual(lora_runtime.lora_patch_count, 1)
                self.assertEqual(base_runtime.snapshot()["reference_count"], 1)
                self.assertEqual(len(built), 1)
                self.assertEqual(built[0][1].item(), 4.0)
                self.assertEqual(built[0][1].dtype, torch.bfloat16)
                self.assertEqual(built[0][2].item(), 0.0)
                self.assertIs(diffusion.xdna_active_runtime, lora_runtime)

                dispatch = loader.CacheBackedDiffusionWrapper()
                executor = types.SimpleNamespace(class_obj=diffusion)
                with mock.patch.object(
                    SharedRuntime, "forward", autospec=True, return_value="output"
                ) as forward:
                    self.assertEqual(dispatch(executor, "x", "t", "c"), "output")
                self.assertIs(forward.call_args.args[0], lora_runtime)

                lora_model.cleanup()
                self.assertIsNone(diffusion.xdna_active_runtime)
                with self.assertRaisesRegex(RuntimeError, "through a ComfyUI sampler"):
                    dispatch(executor, "x", "t", "c")

                lora_model.pre_run()
                self.assertEqual(len(built), 1)
                again = lora_model.clone()
                again.pre_run()
                self.assertEqual(len(built), 1)
                self.assertIs(again.get_attachment(ATTACHMENT_KEY).runtime, lora_runtime)

                suspend.reset_mock()
                base_model.pre_run()
                self.assertEqual(len(built), 1)
                suspend.assert_called_once_with(lora_runtime)
                self.assertIs(diffusion.xdna_active_runtime, base_runtime)

                changed = base_model.clone()
                changed.add_patches({key: adapter}, 1.0)
                with (
                    mock.patch.object(
                        loader, "_file_identity_token", return_value="replaced"
                    ),
                    self.assertRaisesRegex(RuntimeError, "changed after loading"),
                ):
                    changed.pre_run()
                self.assertEqual(len(built), 1)

                unsupported = base_model.clone()
                unsupported.add_patches({key: object()})
                with self.assertRaisesRegex(RuntimeError, "ordinary additive"):
                    unsupported.pre_run()
                self.assertIs(
                    unsupported.get_attachment(ATTACHMENT_KEY).runtime, base_runtime
                )
            for model in (base_model, lora_model, again, changed, unsupported):
                model.get_attachment(ATTACHMENT_KEY).cleanup()
            self.assertEqual(base_runtime.snapshot()["state"], "closed")
            self.assertEqual(lora_runtime.snapshot()["state"], "closed")

    def test_suspend_releases_xrt_session_but_keeps_runtime_prepared(self):
        runtime = SharedRuntime(Path("fixture.safetensors"))
        chain = mock.Mock(is_open=True)
        runtime._runtime = chain
        runtime.diagnostics.state = "open"
        runtime.suspend()
        chain.close_session.assert_called_once_with()
        chain.close.assert_not_called()
        self.assertEqual(runtime.snapshot()["state"], "prepared")
        runtime.close()
        chain.close.assert_called_once_with()

    def test_cache_loader_closes_on_source_change_and_rejects_gpu(self):
        with self.cache_loader_module() as loader:
            loader.folder_paths.get_full_path_or_raise = mock.Mock(return_value="fixture.safetensors")
            with (
                mock.patch.object(loader, "_load_host_model", return_value=FakeBaseModel()),
                mock.patch.object(loader, "_file_identity_token", side_effect=["before", "after"]),
                mock.patch.object(SharedRuntime, "prepare"),
                mock.patch.object(SharedRuntime, "close") as close,
                self.assertRaisesRegex(RuntimeError, "changed while loading"),
            ):
                loader.load_cache_backed_model("fixture.safetensors")
            close.assert_called_once()
            loader.comfy.model_management.get_torch_device.return_value = torch.device("cuda")
            with self.assertRaisesRegex(RuntimeError, "requires ComfyUI --cpu"):
                loader.load_cache_backed_model("fixture.safetensors")

    def test_cache_model_cannot_be_attached_again(self):
        model = FakePatcher()
        model.model.diffusion_model.xdna_cache_backed = True
        with self.assertRaisesRegex(RuntimeError, "already provides an attached MODEL"):
            LoadAttachAnimaXDNAModel().attach(model, AUTO_CHECKPOINT, False)

    def test_cache_loader_node_cache_tracks_source_and_options(self):
        with mock.patch.object(LoadAnimaBF16, "IS_CHANGED", return_value="source-a"):
            initial = LoadAnimaXDNACache.IS_CHANGED("fixture.safetensors", False)
            self.assertEqual(
                initial, LoadAnimaXDNACache.IS_CHANGED("fixture.safetensors", False)
            )
            for options in (
                {"rebuild_cache": True},
                {"rebuild_cache": False, "cache_dir": "different-cache"},
                {"rebuild_cache": False, "qkv_chaining": False},
            ):
                self.assertNotEqual(
                    initial, LoadAnimaXDNACache.IS_CHANGED("fixture.safetensors", **options)
                )
        with mock.patch.object(LoadAnimaBF16, "IS_CHANGED", return_value="source-b"):
            self.assertNotEqual(
                initial, LoadAnimaXDNACache.IS_CHANGED("fixture.safetensors", False)
            )

    def test_attach_clones_model_and_registers_diffusion_wrapper(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "model.safetensors")
            checkpoint.write_bytes(b"fixture")
            source = FakePatcher()
            with mock.patch.object(SharedRuntime, "prepare"):
                attached, status = LoadAttachAnimaXDNAModel().attach(
                    source, str(checkpoint), False
                )
            self.assertIsNot(attached, source)
            self.assertIs(attached.model, source.model)
            sys.modules["comfy.model_management"].load_models_gpu.assert_called_once_with(
                [source]
            )
            self.assertEqual(source.wrappers, {})
            self.assertIn("created", status)
            self.assertIn(
                WRAPPER_KEY,
                attached.wrappers["diffusion_model"],
            )
            attachment = attached.get_attachment(ATTACHMENT_KEY)
            self.assertIsInstance(attachment, RuntimeAttachment)
            attachment.cleanup()

    def test_attach_defaults_to_auto_checkpoint_selection(self):
        checkpoint_input = LoadAttachAnimaXDNAModel.INPUT_TYPES()["required"][
            "checkpoint"
        ]
        self.assertEqual(checkpoint_input[1]["default"], AUTO_CHECKPOINT)

    def test_attach_auto_cache_identity_does_not_require_evaluated_model(self):
        identity = LoadAttachAnimaXDNAModel.IS_CHANGED(
            AUTO_CHECKPOINT, False
        )
        self.assertEqual(
            json.loads(identity)["checkpoint"],
            AUTO_CHECKPOINT,
        )

    def test_attach_auto_uses_loader_provenance(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "model.safetensors")
            checkpoint.write_bytes(b"fixture")
            source = FakePatcher()
            source.set_attachments(
                SOURCE_PROVENANCE_KEY,
                ModelSourceProvenance(
                    selector="diffusion_models:model.safetensors",
                    path=str(checkpoint.resolve()),
                    identity_token=_file_identity_token(checkpoint),
                    loader_total_ms=11.0,
                    loader_model_ms=9.0,
                ),
            )
            cloned = source.clone()
            with mock.patch.object(SharedRuntime, "prepare"):
                attached, status_json = LoadAttachAnimaXDNAModel().attach(
                    cloned, AUTO_CHECKPOINT, False
                )
            attachment = attached.get_attachment(ATTACHMENT_KEY)
            self.assertEqual(attachment.runtime.checkpoint, checkpoint.resolve())
            status = json.loads(status_json)
            self.assertEqual(status["source_loader_total_ms"], 11.0)
            self.assertEqual(status["source_loader_model_ms"], 9.0)
            self.assertGreaterEqual(status["attach_total_ms"], 0.0)
            self.assertGreaterEqual(status["attach_model_load_ms"], 0.0)
            attachment.cleanup()

    def test_attach_auto_requires_loader_provenance(self):
        with self.assertRaisesRegex(
            RuntimeError, "requires a MODEL loaded by Load Anima"
        ):
            LoadAttachAnimaXDNAModel().attach(
                FakePatcher(), AUTO_CHECKPOINT, False
            )

    def test_attach_auto_rejects_checkpoint_changed_after_load(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "model.safetensors")
            checkpoint.write_bytes(b"first")
            source = FakePatcher()
            source.set_attachments(
                SOURCE_PROVENANCE_KEY,
                ModelSourceProvenance(
                    selector="diffusion_models:model.safetensors",
                    path=str(checkpoint.resolve()),
                    identity_token="loaded-identity",
                ),
            )
            with (
                mock.patch(
                    "comfyui_xdna_nodes.model_wrapper._file_identity_token",
                    return_value="changed-identity",
                ),
                self.assertRaisesRegex(
                    RuntimeError, "changed after the MODEL was loaded"
                ),
            ):
                LoadAttachAnimaXDNAModel().attach(
                    source, AUTO_CHECKPOINT, False
                )

    def test_attach_cache_identity_is_stable_until_checkpoint_or_options_change(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "model.safetensors")
            checkpoint.write_bytes(b"first")
            initial = LoadAttachAnimaXDNAModel.IS_CHANGED(
                str(checkpoint), False
            )
            self.assertEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(str(checkpoint), False),
            )
            self.assertNotEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(str(checkpoint), True),
            )
            rebuilt = LoadAttachAnimaXDNAModel.IS_CHANGED(
                str(checkpoint), True
            )
            self.assertEqual(
                rebuilt,
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint), True
                ),
            )
            self.assertNotEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint), False, qkv_chaining=False
                ),
            )
            self.assertNotEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint), False, cache_dir=temporary
                ),
            )
            self.assertEqual(
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint), False, cache_dir=temporary
                ),
                LoadAttachAnimaXDNAModel.IS_CHANGED(
                    str(checkpoint),
                    False,
                    cache_dir=str(Path(temporary, ".")),
                ),
            )
            checkpoint.write_bytes(b"replacement")
            self.assertNotEqual(
                initial,
                LoadAttachAnimaXDNAModel.IS_CHANGED(str(checkpoint), False),
            )

    def test_bf16_loader_cache_identity_tracks_checkpoint_file(self):
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary, "model.safetensors")
            checkpoint.write_bytes(b"first")
            folder_paths = types.ModuleType("folder_paths")
            folder_paths.get_full_path_or_raise = mock.Mock(
                return_value=str(checkpoint)
            )
            with mock.patch.dict("sys.modules", {"folder_paths": folder_paths}):
                initial = LoadAnimaBF16.IS_CHANGED("model.safetensors")
                self.assertEqual(
                    initial,
                    LoadAnimaBF16.IS_CHANGED("model.safetensors"),
                )
                checkpoint.write_bytes(b"replacement")
                self.assertNotEqual(
                    initial,
                    LoadAnimaBF16.IS_CHANGED("model.safetensors"),
                )

    def test_bf16_loader_lists_diffusion_models_and_full_checkpoints(self):
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_filename_list = mock.Mock(
            side_effect=lambda category: {
                "diffusion_models": ["anima.safetensors"],
                "checkpoints": ["wai-nova.safetensors"],
            }[category]
        )
        with mock.patch.dict("sys.modules", {"folder_paths": folder_paths}):
            choices = LoadAnimaBF16.INPUT_TYPES()["required"]["unet_name"][0]
        self.assertEqual(
            choices,
            [
                "diffusion_models:anima.safetensors",
                "checkpoints:wai-nova.safetensors",
            ],
        )

    def test_bf16_loader_resolves_qualified_checkpoint_category(self):
        checkpoint = Path("wai-nova.safetensors")
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_full_path_or_raise = mock.Mock(
            return_value=str(checkpoint)
        )
        with (
            mock.patch.dict("sys.modules", {"folder_paths": folder_paths}),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper._file_identity_token",
                return_value="identity",
            ) as identity,
        ):
            self.assertEqual(
                LoadAnimaBF16.IS_CHANGED(
                    "checkpoints:wai-nova.safetensors"
                ),
                "identity",
            )
        folder_paths.get_full_path_or_raise.assert_called_once_with(
            "checkpoints", "wai-nova.safetensors"
        )
        identity.assert_called_once_with(checkpoint)

    def test_identity_mismatch_closes_before_runtime_open(self):
        provider = mock.Mock()
        runtime = SharedRuntime(
            Path("fixture.safetensors"),
            effective_tensor_provider=provider,
        )
        chain = mock.Mock()
        chain.source_identity = {
            "schema": "native-model-diffusion-model",
            "block_fingerprint": "source",
        }
        chain.execution_identity = {
            "normalization": "BF16",
            "source_dtypes": ["BF16"],
            "block_fingerprint": "source",
        }
        chain.cache_status = None
        with (
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.AnimaXDNAChainRuntime",
                return_value=chain,
            ),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.fingerprint_model_blocks",
                return_value=("model", "comfyui-anima-module"),
            ),
            self.assertRaisesRegex(RuntimeError, "MODEL/checkpoint mismatch"),
        ):
            runtime.prepare(Anima())
        chain.prepare_weight_cache.assert_called_once()
        chain.close.assert_called_once()
        self.assertIsNone(runtime.effective_tensor_provider)
        self.assertIsNone(chain.effective_tensor_provider)

    def test_clones_share_runtime_with_reference_counted_attachments(self):
        runtime = SharedRuntime(Path("fixture.safetensors"))
        first = RuntimeAttachment(runtime)
        second = first.on_model_patcher_clone()
        self.assertEqual(runtime.snapshot()["reference_count"], 2)
        first.cleanup()
        self.assertEqual(runtime.snapshot()["reference_count"], 1)
        second.cleanup()
        self.assertEqual(runtime.snapshot()["state"], "closed")

    def test_attachment_finalizer_releases_reference(self):
        runtime = SharedRuntime(Path("fixture.safetensors"))
        attachment = RuntimeAttachment(runtime)
        del attachment
        gc.collect()
        self.assertEqual(runtime.snapshot()["state"], "closed")

    def test_supports_ordinary_lora_without_mutating_shared_parameters(self):
        patcher = FakePatcher()
        parameters = fake_anima_parameters(torch.bfloat16)
        parameters["blocks.0.self_attn.q_proj.weight"] = torch.nn.Parameter(
            torch.ones((1, 1), dtype=torch.bfloat16)
        )
        patcher.model.diffusion_model = FakeAnimaParameters(parameters)
        key = "diffusion_model.blocks.0.self_attn.q_proj.weight"
        first = self.LoRAAdapter(
            set(),
            (
                torch.tensor([[2.0]]),
                torch.tensor([[3.0]]),
                None,
                None,
                None,
                None,
            ),
        )
        second = self.LoRAAdapter(
            set(),
            (
                torch.tensor([[4.0]]),
                torch.tensor([[5.0]]),
                None,
                None,
                None,
                None,
            ),
        )
        patcher.patches[key] = [
            (0.5, first, 1.0, None, None),
            (0.25, second, 1.0, None, None),
        ]
        provider, patch_count, base_fingerprint, base_schema = (
            _effective_lora_provider(patcher)
        )
        effective = provider("transformer_blocks.0.attn1.to_q.weight")
        snapshot_fingerprint = provider.input_fingerprint
        first.weights[0].fill_(100.0)
        first.weights[1].fill_(100.0)
        second.weights[0].fill_(100.0)
        second.weights[1].fill_(100.0)
        effective_after_mutation = provider(
            "transformer_blocks.0.attn1.to_q.weight"
        )
        self.assertEqual(patch_count, 2)
        self.assertIsInstance(base_fingerprint, str)
        self.assertEqual(base_schema, "comfyui-model-wrapper")
        self.assertEqual(effective.item(), 9.0)
        self.assertEqual(effective_after_mutation.item(), 9.0)
        self.assertEqual(provider.input_fingerprint, snapshot_fingerprint)
        self.assertEqual(provider.snapshot_tensor_count, 4)
        self.assertEqual(provider.snapshot_bytes, 16)
        self.assertGreaterEqual(provider.snapshot_total_ms, 0.0)
        self.assertGreaterEqual(provider.snapshot_clone_ms, 0.0)
        self.assertGreaterEqual(provider.snapshot_hash_ms, 0.0)
        self.assertGreaterEqual(provider.base_identity_ms, 0.0)
        self.assertEqual(
            parameters["blocks.0.self_attn.q_proj.weight"].item(),
            1.0,
        )

    def test_rejects_unsupported_lora_and_transformer_patches(self):
        patcher = FakePatcher()
        patcher.model.diffusion_model = FakeAnimaParameters(
            fake_anima_parameters(torch.bfloat16)
        )
        key = "diffusion_model.blocks.0.self_attn.q_proj.weight"
        patcher.patches[key] = [(1.0, object(), 1.0, None, None)]
        with self.assertRaisesRegex(RuntimeError, "ordinary additive"):
            _effective_lora_provider(patcher)
        patcher.patches.clear()
        patcher.model_options["transformer_options"]["patches"] = {
            "mlp_patch": [object()]
        }
        with self.assertRaisesRegex(RuntimeError, "transformer patches"):
            _validate_patcher(patcher)

    def test_supports_turbo_lora_attention_norm_diff(self):
        patcher = FakePatcher()
        parameters = fake_anima_parameters(torch.bfloat16)
        parameters["blocks.0.self_attn.q_norm.weight"] = torch.nn.Parameter(
            torch.ones(2, dtype=torch.bfloat16)
        )
        patcher.model.diffusion_model = FakeAnimaParameters(parameters)
        key = "diffusion_model.blocks.0.self_attn.q_norm.weight"
        diff = torch.tensor([0.5, -0.25])
        patcher.patches[key] = [
            (
                0.5,
                ("diff", (diff,)),
                1.0,
                None,
                None,
            )
        ]

        provider, patch_count, _, _ = _effective_lora_provider(patcher)
        effective = provider("transformer_blocks.0.attn1.norm_q.weight")
        diff.zero_()
        snapshotted = provider("transformer_blocks.0.attn1.norm_q.weight")

        self.assertEqual(patch_count, 1)
        torch.testing.assert_close(
            effective.float(),
            torch.tensor([1.25, 0.875]),
        )
        torch.testing.assert_close(snapshotted.float(), effective.float())
        torch.testing.assert_close(
            parameters["blocks.0.self_attn.q_norm.weight"].float(),
            torch.ones(2),
        )

    def test_rejects_unsafe_turbo_lora_diff_patches(self):
        patcher = FakePatcher()
        parameters = fake_anima_parameters(torch.bfloat16)
        patcher.model.diffusion_model = FakeAnimaParameters(parameters)
        projection = "diffusion_model.blocks.0.self_attn.q_proj.weight"
        patcher.patches[projection] = [
            (1.0, ("diff", (torch.zeros(1),)), 1.0, None, None)
        ]
        with self.assertRaisesRegex(RuntimeError, "only for attention q/k norm"):
            _effective_lora_provider(patcher)

        patcher.patches.clear()
        norm = "diffusion_model.blocks.0.self_attn.q_norm.weight"
        patcher.patches[norm] = [
            (1.0, ("diff", (torch.tensor([float("nan")]),)), 1.0, None, None)
        ]
        with self.assertRaisesRegex(RuntimeError, "finite, exact-shape"):
            _effective_lora_provider(patcher)

    def test_status_reports_not_attached(self):
        model, status = AnimaXDNARuntimeStatus().status(FakePatcher())
        self.assertIsInstance(model, FakePatcher)
        self.assertIn("not attached", status)

    def test_runtime_snapshot_names_host_visible_output_bytes(self):
        with TemporaryDirectory() as directory:
            def provider(_key):
                return torch.zeros(1)

            provider.snapshot_total_ms = 12.0
            provider.snapshot_clone_ms = 3.0
            provider.snapshot_hash_ms = 2.0
            provider.base_identity_ms = 6.0
            provider.snapshot_bytes = 1024
            provider.snapshot_tensor_count = 4
            runtime = SharedRuntime(
                Path(directory) / "checkpoint.safetensors",
                effective_tensor_provider=provider,
                attach_model_load_ms=7.0,
            )
            snapshot = runtime.snapshot()
            self.assertEqual(snapshot["last_d2h_host_visible_bytes"], 0)
            self.assertEqual(snapshot["attach_model_load_ms"], 7.0)
            self.assertEqual(snapshot["lora_snapshot_total_ms"], 12.0)
            self.assertEqual(snapshot["lora_snapshot_clone_ms"], 3.0)
            self.assertEqual(snapshot["lora_snapshot_hash_ms"], 2.0)
            self.assertEqual(snapshot["lora_base_identity_ms"], 6.0)
            self.assertEqual(snapshot["lora_snapshot_bytes"], 1024)
            self.assertEqual(snapshot["lora_snapshot_tensor_count"], 4)

    def test_model_storage_profile_separates_blocks_from_nonblock_parameters(self):
        parameters = fake_anima_parameters(torch.bfloat16)
        profile = _model_storage_profile(FakeAnimaParameters(parameters))
        self.assertEqual(profile["block_parameter_count"], 560)
        self.assertEqual(profile["block_dtype_numel"], {"bfloat16": 560})
        self.assertEqual(profile["block_parameter_bytes"], 1120)
        self.assertEqual(profile["block_unique_storage_bytes"], 1120)
        self.assertEqual(profile["nonblock_parameter_bytes"], 2)

    def test_fp32_block_memory_is_reported_without_mutating_parameters(self):
        parameters = fake_anima_parameters(torch.float32)
        diffusion_model = FakeAnimaParameters(parameters)
        runtime = SharedRuntime(Path("fixture.safetensors"))
        chain = mock.Mock()
        chain.source_identity = {
            "schema": "native-model-diffusion-model",
            "block_fingerprint": "same",
        }
        chain.execution_identity = {
            "normalization": "BF16",
            "source_dtypes": ["BF16"],
            "block_fingerprint": "same",
        }
        chain.cache_status = None
        with (
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.AnimaXDNAChainRuntime",
                return_value=chain,
            ),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.fingerprint_model_blocks",
                return_value=("same", "comfyui-anima-module"),
            ),
        ):
            runtime.prepare(diffusion_model)
        self.assertEqual(runtime.snapshot()["model_block_parameter_bytes"], 2240)
        self.assertIn(
            "Load Anima (BF16)",
            runtime.snapshot()["model_fp32_block_memory_warning"],
        )
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in parameters.values())
        )
        runtime.close()

    def test_f16_source_reports_bf16_packed_cache_guidance(self):
        parameters = fake_anima_parameters(torch.bfloat16)
        diffusion_model = FakeAnimaParameters(parameters)
        runtime = SharedRuntime(Path("fixture.safetensors"))
        chain = mock.Mock()
        chain.source_identity = {
            "schema": "native-model-diffusion-model",
            "block_fingerprint": "raw-f16",
        }
        chain.execution_identity = {
            "normalization": "BF16",
            "source_dtypes": ["F16"],
            "block_fingerprint": "normalized",
        }
        chain.cache_status = None
        with (
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.AnimaXDNAChainRuntime",
                return_value=chain,
            ),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper.fingerprint_model_blocks",
                return_value=("normalized", "comfyui-anima-module"),
            ),
            mock.patch("builtins.print") as output,
        ):
            runtime.prepare(diffusion_model)
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["source_fingerprint"], "raw-f16")
        self.assertEqual(
            snapshot["source_execution_fingerprint"], "normalized"
        )
        self.assertIn(
            "original checkpoint is unchanged",
            snapshot["source_normalization_message"],
        )
        self.assertIn(
            "future runs reuse this cache",
            snapshot["source_normalization_message"],
        )
        output.assert_called_once_with(
            "[Anima XDNA] " + snapshot["source_normalization_message"]
        )
        runtime.close()

    def test_bf16_loader_passes_explicit_dtype_to_comfyui(self):
        model = FakePatcher()
        model.model.diffusion_model = FakeAnimaParameters(
            fake_anima_parameters(torch.bfloat16)
        )
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_diffusion_model = mock.Mock(return_value=model)
        comfy = types.ModuleType("comfy")
        comfy.sd = comfy_sd
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_full_path_or_raise = mock.Mock(
            return_value="anima.safetensors"
        )
        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "comfy": comfy,
                    "comfy.sd": comfy_sd,
                    "folder_paths": folder_paths,
                },
            ),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper._file_identity_token",
                return_value="source-identity",
            ),
        ):
            (loaded,) = LoadAnimaBF16().load("anima.safetensors")
        self.assertIs(loaded, model)
        provenance = loaded.get_attachment(SOURCE_PROVENANCE_KEY)
        self.assertIsInstance(provenance, ModelSourceProvenance)
        self.assertEqual(
            provenance.selector,
            "anima.safetensors",
        )
        self.assertEqual(provenance.identity_token, "source-identity")
        self.assertGreaterEqual(provenance.loader_total_ms, 0.0)
        self.assertGreaterEqual(provenance.loader_model_ms, 0.0)
        comfy_sd.load_diffusion_model.assert_called_once_with(
            "anima.safetensors",
            model_options={"dtype": torch.bfloat16},
        )

    def test_bf16_loader_loads_qualified_full_checkpoint(self):
        model = FakePatcher()
        model.model.diffusion_model = FakeAnimaParameters(
            fake_anima_parameters(torch.bfloat16)
        )
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_diffusion_model = mock.Mock(return_value=model)
        comfy = types.ModuleType("comfy")
        comfy.sd = comfy_sd
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_full_path_or_raise = mock.Mock(
            return_value="wai-nova.safetensors"
        )
        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "comfy": comfy,
                    "comfy.sd": comfy_sd,
                    "folder_paths": folder_paths,
                },
            ),
            mock.patch(
                "comfyui_xdna_nodes.model_wrapper._file_identity_token",
                return_value="source-identity",
            ),
        ):
            (loaded,) = LoadAnimaBF16().load(
                "checkpoints:wai-nova.safetensors"
            )
        self.assertIs(loaded, model)
        folder_paths.get_full_path_or_raise.assert_called_once_with(
            "checkpoints", "wai-nova.safetensors"
        )
        comfy_sd.load_diffusion_model.assert_called_once_with(
            "wai-nova.safetensors",
            model_options={"dtype": torch.bfloat16},
        )

    def test_bf16_loader_rejects_dtype_expansion(self):
        model = FakePatcher()
        model.model.diffusion_model = FakeAnimaParameters(
            fake_anima_parameters(torch.float32)
        )
        comfy_sd = types.ModuleType("comfy.sd")
        comfy_sd.load_diffusion_model = mock.Mock(return_value=model)
        comfy = types.ModuleType("comfy")
        comfy.sd = comfy_sd
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_full_path_or_raise = mock.Mock(
            return_value="anima.safetensors"
        )
        with (
            mock.patch.dict(
                "sys.modules",
                {"comfy": comfy, "comfy.sd": comfy_sd, "folder_paths": folder_paths},
            ),
            self.assertRaisesRegex(RuntimeError, "refusing to load a memory-expanded"),
        ):
            LoadAnimaBF16().load("anima.safetensors")

    def test_forward_maps_1024_latent_to_4096_token_chain(self):
        from types import SimpleNamespace

        from anima_xdna_poc.block import rotary_embedding
        from anima_xdna_poc.errors import UnsupportedTensor

        seen = {}

        class FakeAnima:
            blocks = [None] * 28
            t_embedding_norm = staticmethod(lambda value: value)
            t_embedder = (
                lambda tokens: tokens.float().unsqueeze(-1).expand(-1, -1, 2048),
                lambda value: (value, torch.zeros(*value.shape[:2], 6144)),
            )

            def prepare_embedded_sequence(self, x, fps=None, padding_mask=None):
                grid = (x.shape[0], 1, x.shape[-2] // 2, x.shape[-1] // 2, 2048)
                return torch.randn(grid), None, None

            def final_layer(self, residual, embedded_timestep, adaln_lora_B_T_3D):
                seen["residual"] = residual
                return residual

            def unpatchify(self, patches):
                batch, _, height, width, _ = patches.shape
                return torch.zeros(batch, 16, 1, 2 * height, 2 * width)

        FakeAnima.__name__ = "Anima"
        FakeAnima.__module__ = "comfy.ldm.anima.model"
        config = SimpleNamespace(head_dim=128, rope_scale=(1.0, 4.0, 4.0))
        chain_runtime = mock.Mock(config=config, cache_status=None)
        chain_runtime.run_range.side_effect = lambda inputs: SimpleNamespace(
            output=inputs.hidden_states + 1,
            blocks=(),
            wall_ms=0.0,
            dispatch_count=0,
        )
        runtime = SharedRuntime(Path("fixture.safetensors"))
        model = FakeAnima()
        context = torch.zeros(2, 512, 1024)
        with mock.patch.object(
            runtime, "open", return_value=chain_runtime
        ), mock.patch.object(runtime, "_interrupt"):
            output = runtime.forward(
                model, torch.zeros(2, 16, 1, 128, 128), torch.ones(2), context
            )
            with self.assertRaisesRegex(UnsupportedTensor, "1024x1024"):
                runtime.forward(
                    model, torch.zeros(1, 16, 1, 96, 96), torch.ones(1), context[:1]
                )
        self.assertEqual(tuple(output.shape), (2, 16, 1, 128, 128))
        calls = chain_runtime.run_range.call_args_list
        self.assertEqual(len(calls), 2)
        inputs = calls[0].args[0]
        self.assertEqual(tuple(inputs.hidden_states.shape), (1, 4096, 2048))
        expected_cos, expected_sin = rotary_embedding(config, 4096)
        self.assertTrue(torch.equal(inputs.image_rotary_emb[0], expected_cos))
        self.assertTrue(torch.equal(inputs.image_rotary_emb[1], expected_sin))
        residual = seen["residual"]
        self.assertEqual(tuple(residual.shape), (2, 1, 64, 64, 2048))
        self.assertTrue(
            torch.equal(
                residual[0].reshape(1, 4096, 2048).to(torch.bfloat16),
                inputs.hidden_states + 1,
            )
        )

    def test_rejects_non_anima_model(self):
        patcher = FakePatcher()
        patcher.model.diffusion_model = object()
        with self.assertRaisesRegex(RuntimeError, "Anima MODEL"):
            _validate_patcher(patcher)


if __name__ == "__main__":
    unittest.main()
