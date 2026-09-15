from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from typing import Any
import torch.nn as nn
from pathlib import Path
from typing import Any
from collections import Counter

import torch
import json
from typing import Optional
from typing import Dict


class _PassthroughCallable:
    """Small callable used to replace activation attributes without registering as a submodule.

    Using a plain callable avoids adding an nn.Module (like Identity) to the model's _modules
    registry while still providing an attribute that returns its input when called (so code
    that does `self.softmax(x)` continues to work during tracing/export).
    """
    def __call__(self, x):
        return x


def _ensure_project_importable() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_if_zip(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_dir = tempfile.TemporaryDirectory(prefix="evodenss_export_")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_dir.name)
        # Prefer an inner "run9" directory if present inside the extracted archive.
        extracted_root = Path(temp_dir.name)
        # search for directories named like run9, run_9, run-9 (case-insensitive)
        run_candidates = [
            p for p in extracted_root.rglob("*") if p.is_dir() and p.name.lower() in ("run9",)
        ]
        if not run_candidates:
            # try some common variants
            for p in extracted_root.rglob("*"):
                if p.is_dir() and p.name.lower().replace('_', '').replace('-', '') == 'run9':
                    run_candidates.append(p)
                    break

        if run_candidates:
            # pick the first match (should be the intended run9 folder)
            return run_candidates[0], temp_dir

        return extracted_root, temp_dir
    return input_path, None


def _find_artifacts(search_root: Path) -> tuple[Path | None, Path]:
    preferred_pairs = [
        ("complete_model.pt", "complete_weights.pt"),
        ("model.pt", "weights.pt"),
        ("extended_complete_model.pt", "extended_complete_weights.pt"),
    ]

    for model_name, weights_name in preferred_pairs:
        model_matches = list(search_root.rglob(model_name))
        for model_path in model_matches:
            weights_path = model_path.parent / weights_name
            if weights_path.exists():
                return model_path, weights_path

    model_candidates = [
        p
        for p in search_root.rglob("*.pt")
        if "model" in p.name.lower() and "weights" not in p.name.lower()
    ] + [
        p
        for p in search_root.rglob("*.pth")
        if "model" in p.name.lower() and "weights" not in p.name.lower()
    ]

    weight_candidates = [
        p
        for p in search_root.rglob("*.pt")
        if "weight" in p.name.lower() or "state_dict" in p.name.lower()
    ] + [
        p
        for p in search_root.rglob("*.pth")
        if "weight" in p.name.lower() or "state_dict" in p.name.lower()
    ]

    for model_path in model_candidates:
        for weight_path in weight_candidates:
            if model_path.parent == weight_path.parent:
                return model_path, weight_path

    if weight_candidates:
        return None, weight_candidates[0]

    raise FileNotFoundError(
        "Could not automatically find weights file. "
        "Pass --weights-path explicitly."
    )


def _extract_state_dict(raw_weights: Any) -> Mapping[str, Any]:
    if isinstance(raw_weights, Mapping):
        if "state_dict" in raw_weights and isinstance(raw_weights["state_dict"], Mapping):
            return raw_weights["state_dict"]
        if all(isinstance(k, str) for k in raw_weights.keys()):
            return raw_weights
    raise ValueError("Weights file does not contain a valid state_dict mapping.")


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _try_load_model(model_path: Path | None) -> Any | None:
    if model_path is None:
        return None
    try:
        return _torch_load(model_path)
    except ModuleNotFoundError as exc:
        print(
            "Warning: could not deserialize full model due to missing module during unpickle: "
            f"{exc}. Continuing with pure state_dict export only."
        )
        return None


def _parse_trace_shape(shape_text: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --trace-shape value: {shape_text}") from exc
    if len(dims) < 1:
        raise ValueError("--trace-shape must contain at least one dimension, e.g. 3,32,32")
    return dims


def _export_torchscript(
    output_dir: Path,
    export_name: str,
    model_obj: Any | None,
    method: str,
    trace_shape: str,
    trace_batch_size: int,
) -> None:
    if model_obj is None:
        print("Warning: TorchScript export skipped because full model could not be deserialized.")
        return
    if not hasattr(model_obj, "eval"):
        print("Warning: TorchScript export skipped because loaded model is not a torch.nn.Module-like object.")
        return

    model_obj.eval()
    ts_path = output_dir / f"{export_name}_torchscript.ts"

    scripted_model: Any
    if method == "script":
        scripted_model = torch.jit.script(model_obj)
    elif method == "trace":
        input_shape = (trace_batch_size, *_parse_trace_shape(trace_shape))
        example_input = torch.randn(*input_shape)
        scripted_model = torch.jit.trace(model_obj, example_input)
    else:
        raise ValueError(f"Unsupported torchscript method: {method}")

    torch.jit.save(scripted_model, str(ts_path))
    print(f"Saved TorchScript: {ts_path}")


def _export_onnx(
    output_dir: Path,
    export_name: str,
    model_obj: Any | None,
    trace_shape: str,
    trace_batch_size: int,
    onnx_opset: int,
    onnx_dynamic: bool,
    input_names: list[str],
    output_names: list[str],
) -> None:
    if model_obj is None:
        print("Warning: ONNX export skipped because full model could not be deserialized.")
        return
    if not hasattr(model_obj, "eval"):
        print("Warning: ONNX export skipped because loaded model is not a torch.nn.Module-like object.")
        return

    model_obj.eval()
    onnx_path = output_dir / f"{export_name}.onnx"

    input_shape = (trace_batch_size, *_parse_trace_shape(trace_shape))
    example_input = torch.randn(*input_shape)

    dynamic_axes = None
    if onnx_dynamic:
        dynamic_axes = {name: {0: "batch_size"} for name in input_names + output_names}

    try:
        torch.onnx.export(
            model_obj,
            example_input,
            str(onnx_path),
            opset_version=onnx_opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        print(f"Saved ONNX: {onnx_path}")
    except Exception as exc:
        print(f"ONNX export failed: {type(exc).__name__}: {exc}")


def _describe_model(model_obj: Any | None, state_dict: Mapping[str, Any]) -> None:
    """Print a concise summary of modules/types and state-dict contents."""
    print("\n=== Model Summary ===")
    if model_obj is None:
        print("Full model object not deserializable; showing state_dict summary only.")
    else:
        try:
            type_counts = Counter()
            for name, mod in model_obj.named_modules():
                type_counts[mod.__class__.__name__] += 1
            print("Module types (top 10):")
            for t, c in type_counts.most_common(10):
                print(f"  {t}: {c}")
        except Exception as exc:
            print(f"Could not enumerate model modules: {type(exc).__name__}: {exc}")

    # State-dict summary
    try:
        param_count = 0
        tensor_entries = []
        for k, v in state_dict.items():
            if hasattr(v, "numel"):
                param_count += int(v.numel())
                tensor_entries.append((k, tuple(v.shape)))
            else:
                tensor_entries.append((k, type(v).__name__))

        print(f"State-dict entries: {len(tensor_entries)}; total parameters: {param_count}")
        print("First 20 state-dict keys/shapes:")
        for k, s in tensor_entries[:20]:
            print(f"  {k}: {s}")
    except Exception as exc:
        print(f"Could not inspect state_dict: {type(exc).__name__}: {exc}")
    print("=== End Model Summary ===\n")


def _dump_layer_info(model_obj: Any | None, example_input: torch.Tensor | None = None) -> None:
    """Print each named module, its type, its parameter shapes and (if example_input provided) its output shape."""
    print("\n=== Layer-by-layer details ===")
    if model_obj is None:
        print("Full model object not deserializable; cannot dump layer-by-layer module info.")
        return

    # gather parameter info per module
    try:
        for name, module in model_obj.named_modules():
            if name == "":
                continue
            print(f"Module: {name} | Type: {module.__class__.__name__}")
            # params belonging directly to this module
            param_list = list(module.named_parameters(recurse=False))
            if param_list:
                for pname, p in param_list:
                    print(f"  Param: {pname} shape={tuple(p.shape)} dtype={p.dtype}")
            else:
                print("  (no parameters)")
    except Exception as exc:
        print(f"Could not enumerate module parameters: {type(exc).__name__}: {exc}")

    # optionally run a forward to capture output shapes
    if example_input is not None:
        print("\nRunning a single forward pass to capture output shapes (cpu)")
        hooks = []
        outputs: Dict[str, tuple[int, ...] | str] = {}

        def make_hook(n: str):
            def hook(module, inp, out):
                try:
                    if isinstance(out, torch.Tensor):
                        outputs[n] = tuple(out.shape)
                    elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                        outputs[n] = tuple(out[0].shape)
                    else:
                        outputs[n] = str(type(out))
                except Exception as e:
                    outputs[n] = f"error: {e}"
            return hook

        try:
            for name, module in model_obj.named_modules():
                if name == "":
                    continue
                try:
                    h = module.register_forward_hook(make_hook(name))
                    hooks.append(h)
                except Exception:
                    pass

            model_obj.eval()
            with torch.no_grad():
                _ = model_obj(example_input)

            # print captured outputs
            for name, shape in outputs.items():
                print(f"Output - Module: {name} -> {shape}")
        except Exception as exc:
            print(f"Forward pass failed: {type(exc).__name__}: {exc}")
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

    print("=== End Layer details ===\n")


def _collect_layer_info(model_obj: Any | None, example_input: Optional[torch.Tensor] = None) -> dict:
    """Return a serializable dict with module types, param shapes and optional output shapes."""
    info: dict = {"modules": {}, "forward_example_input_shape": None}
    if model_obj is None:
        return info

    try:
        for name, module in model_obj.named_modules():
            if name == "":
                continue
            entry: dict = {"type": module.__class__.__name__, "parameters": []}
            for pname, p in module.named_parameters(recurse=False):
                entry["parameters"].append({"name": pname, "shape": list(p.shape)})
            info["modules"][name] = entry
    except Exception:
        pass

    if example_input is not None:
        info["forward_example_input_shape"] = list(example_input.shape)
        outputs: dict = {}

        def make_hook(n: str):
            def hook(module, inp, out):
                try:
                    if isinstance(out, torch.Tensor):
                        outputs[n] = list(out.shape)
                    elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                        outputs[n] = list(out[0].shape)
                    else:
                        outputs[n] = str(type(out))
                except Exception as e:
                    outputs[n] = f"error: {e}"
            return hook

        hooks = []
        try:
            for name, module in model_obj.named_modules():
                if name == "":
                    continue
                try:
                    h = module.register_forward_hook(make_hook(name))
                    hooks.append(h)
                except Exception:
                    pass

            model_obj.eval()
            with torch.no_grad():
                _ = model_obj(example_input)

            for n, s in outputs.items():
                if n in info["modules"]:
                    info["modules"][n]["output_shape"] = s
                else:
                    info["modules"][n] = {"type": "<unknown>", "parameters": [], "output_shape": s}
        except Exception:
            pass
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

    return info


def _save_exports(
    state_dict: Mapping[str, Any],
    output_dir: Path,
    export_name: str,
    source_model_path: Path | None,
    source_weights_path: Path,
    model_obj: Any | None,
    save_full_model: bool,
    export_torchscript: bool,
    torchscript_method: str,
    trace_shape: str,
    trace_batch_size: int,
    export_onnx: bool=False,
    onnx_opset: int=13,
    onnx_dynamic: bool=False,
    onnx_input_names: list[str]|None=None,
    onnx_output_names: list[str]|None=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_only_path = output_dir / f"{export_name}_weights_only.pth"
    state_dict_path = output_dir / f"{export_name}_state_dict.pth"
    checkpoint_path = output_dir / f"{export_name}_checkpoint.pt"

    torch.save(dict(state_dict), weights_only_path)
    torch.save(dict(state_dict), state_dict_path)
    torch.save(
        {
            "state_dict": dict(state_dict),
            "source_model_file": str(source_model_path) if source_model_path is not None else None,
            "source_weights_file": str(source_weights_path),
            "model_class": model_obj.__class__.__name__ if model_obj is not None else None,
        },
        checkpoint_path,
    )

    print(f"Saved weights-only: {weights_only_path}")
    print(f"Saved state dict: {state_dict_path}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print("Use *_weights_only.pth in other frameworks. Avoid *_full_model.pt outside EvoDENSS.")

    if save_full_model and model_obj is not None:
        full_model_path = output_dir / f"{export_name}_full_model.pt"
        torch.save(model_obj, full_model_path)
        print(f"Saved full model: {full_model_path}")
    elif save_full_model and model_obj is None:
        print("Warning: --save-full-model requested but full model could not be loaded; skipped.")

    if export_torchscript:
        _export_torchscript(
            output_dir=output_dir,
            export_name=export_name,
            model_obj=model_obj,
            method=torchscript_method,
            trace_shape=trace_shape,
            trace_batch_size=trace_batch_size,
        )

    if export_onnx:
        _export_onnx(
            output_dir=output_dir,
            export_name=export_name,
            model_obj=model_obj,
            trace_shape=trace_shape,
            trace_batch_size=trace_batch_size,
            onnx_opset=onnx_opset,
            onnx_dynamic=onnx_dynamic,
            input_names=onnx_input_names or ["input"],
            output_names=onnx_output_names or ["output"],
        )


def _find_final_linear_in_state(state_dict: Mapping[str, Any]) -> tuple[str, str] | None:
    # return (weight_key, bias_key) if found
    candidates: list[tuple[str, str]] = []
    for k in state_dict.keys():
        if k.endswith(".weight") and getattr(state_dict[k], "ndim", None) == 2:
            base = k[:-7]
            bias = base + ".bias"
            if bias in state_dict and getattr(state_dict[bias], "ndim", None) == 1:
                candidates.append((k, bias))

    if not candidates:
        return None

    # priority by keywords
    priority_keys = ["classifier", "fc", "head", "linear", "proj", "projector", "output"]
    for wk, bk in candidates:
        if any(p in wk.lower() or p in bk.lower() for p in priority_keys):
            return wk, bk

    # fallback: pick the candidate whose key appears last alphabetically (likely final)
    candidates.sort()
    return candidates[-1]


def _build_linear_module_from_state(state_dict: Mapping[str, Any]) -> tuple[nn.Module | None, str | None]:
    found = _find_final_linear_in_state(state_dict)
    if found is None:
        return None, None
    wkey, bkey = found
    w = state_dict[wkey]
    b = state_dict[bkey]
    in_features = int(w.shape[1])
    out_features = int(w.shape[0])
    linear = nn.Linear(in_features, out_features)
    # load weights (state_dict expects 'weight' and 'bias')
    linear_state = {"weight": w.clone(), "bias": b.clone()}
    linear.load_state_dict(linear_state)
    return linear, wkey[:-7]


def main() -> None:
    _ensure_project_importable()

    parser = argparse.ArgumentParser(
        description="Export EvoDENSS model artifacts to portable PyTorch .pth/.pt files"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to EvoDENSS folder or .zip containing model artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/exports"),
        help="Directory where exported files are saved",
    )
    parser.add_argument(
        "--export-name",
        type=str,
        default=None,
        help="Prefix for exported filenames (default: input filename stem)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit model file path inside the extracted/input folder",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional explicit weights file path inside the extracted/input folder",
    )
    parser.add_argument(
        "--save-full-model",
        action="store_true",
        help="Also save a pickled full-model .pt file (not portable across frameworks)",
    )
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="Export a TorchScript file (.ts) that can be loaded without evodenss source code",
    )
    parser.add_argument(
        "--torchscript-method",
        choices=["script", "trace"],
        default="script",
        help="TorchScript export method (default: script)",
    )
    parser.add_argument(
        "--trace-shape",
        type=str,
        default="3,32,32",
        help="Input shape C,H,W for trace mode (default: 3,32,32)",
    )
    parser.add_argument(
        "--trace-batch-size",
        type=int,
        default=1,
        help="Batch size for trace mode example input (default: 1)",
    )
    parser.add_argument(
        "--remove-final-softmax",
        action="store_true",
        help="If present, detect and remove final Softmax/LogSoftmax module from the model before export",
    )
    parser.add_argument(
        "--layer-json",
        type=Path,
        default=None,
        help="Optional path to save layer-by-layer JSON description",
    )
    parser.add_argument(
        "--export-linear-only",
        action="store_true",
        help="Export only the final linear layer (built from weights) instead of the full model",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export an ONNX model (.onnx) that can be loaded without evodenss source code",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    parser.add_argument(
        "--onnx-dynamic",
        action="store_true",
        help="Make the batch dimension dynamic in the exported ONNX model",
    )
    parser.add_argument(
        "--onnx-input-names",
        type=str,
        default="input",
        help="Comma-separated input names for ONNX export (default: input)",
    )
    parser.add_argument(
        "--onnx-output-names",
        type=str,
        default="output",
        help="Comma-separated output names for ONNX export (default: output)",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path not found: {args.input_path}")

    resolved_root, temp_dir = _extract_if_zip(args.input_path)

    try:
        if args.model_path is not None and args.weights_path is not None:
            model_path = (
                (resolved_root / args.model_path).resolve()
                if not args.model_path.is_absolute()
                else args.model_path
            )
            weights_path = (
                (resolved_root / args.weights_path).resolve()
                if not args.weights_path.is_absolute()
                else args.weights_path
            )
            if not model_path.exists() or not weights_path.exists():
                raise FileNotFoundError(
                    f"Explicit model/weights paths not found. model={model_path}, weights={weights_path}"
                )
        elif args.model_path is None and args.weights_path is None:
            model_path, weights_path = _find_artifacts(resolved_root)
        elif args.model_path is None and args.weights_path is not None:
            model_path = None
            weights_path = (
                (resolved_root / args.weights_path).resolve()
                if not args.weights_path.is_absolute()
                else args.weights_path
            )
            if not weights_path.exists():
                raise FileNotFoundError(f"Explicit weights path not found: {weights_path}")
        else:
            raise ValueError(
                "Provide either only --weights-path, or both --model-path and --weights-path, or neither."
            )

        if model_path is not None:
            print(f"Using model: {model_path}")
        else:
            print("Using model: <not required>")
        print(f"Using weights: {weights_path}")

        raw_weights = _torch_load(weights_path)
        state_dict = _strip_module_prefix(_extract_state_dict(raw_weights))
        model_obj = _try_load_model(model_path)

        if model_obj is not None and hasattr(model_obj, "load_state_dict"):
            load_result = model_obj.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                print(f"Warning: missing keys when loading weights: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Warning: unexpected keys when loading weights: {load_result.unexpected_keys}")

        # Print model / state-dict summary for user
        _describe_model(model_obj, state_dict)

        # Optionally remove final softmax module
        if args.remove_final_softmax and model_obj is not None:
            def _try_remove_module(parent, name) -> bool:
                """Attempt to remove a child module from parent. Return True if removed."""
                try:
                    # Prefer removing from parent's registered modules dict
                    mods = getattr(parent, "_modules", None)
                    if mods is not None and name in mods:
                        # remove registration
                        mods.pop(name, None)
                        # For Sequential parents we must leave an nn.Module in place; replace with Identity
                        if isinstance(parent, nn.Sequential):
                            try:
                                mods[name] = nn.Identity()
                                return True
                            except Exception:
                                pass
                        # For non-module parents, set a simple callable attribute so self.softmax(...) still works
                        try:
                            setattr(parent, name, _PassthroughCallable())
                        except Exception:
                            # best-effort: try deleting attribute if setting fails
                            try:
                                if hasattr(parent, name):
                                    delattr(parent, name)
                            except Exception:
                                pass
                        return True

                    # If name is numeric (Sequential index), try removing that key
                    try:
                        idx = int(name)
                        key = str(idx)
                        if mods is not None and key in mods:
                            mods.pop(key, None)
                            try:
                                if hasattr(parent, key):
                                    delattr(parent, key)
                            except Exception:
                                pass
                            return True
                    except Exception:
                        pass

                    return False
                except Exception:
                    return False

            softmax_candidates = []
            for mod_name, mod in model_obj.named_modules():
                if isinstance(mod, (nn.Softmax, nn.LogSoftmax)):
                    softmax_candidates.append(mod_name)

            if not softmax_candidates:
                print("No Softmax/LogSoftmax module found to remove (softmax may be applied in forward()).")
            else:
                # pick deepest candidate (longest dotted path)
                softmax_candidates.sort(key=lambda s: (s.count('.'), len(s)))
                target = softmax_candidates[-1]
                parts = target.split('.') if target else []
                parent = model_obj
                for p in parts[:-1]:
                    # attribute may be numeric for Sequential
                    if hasattr(parent, p):
                        parent = getattr(parent, p)
                    else:
                        try:
                            parent = parent[int(p)]
                        except Exception:
                            parent = None
                            break
                if parent is None:
                    print(f"Could not locate parent for module '{target}'; skipping softmax removal.")
                else:
                    last = parts[-1] if parts else None
                    if last is None:
                        print("Unexpected module path; skipping.")
                    else:
                        removed = _try_remove_module(parent, last)
                        if removed:
                            print(f"Removed Softmax module '{target}' from model (kept passthrough attribute where applicable).")
                        else:
                            # fallback: keep a passthrough callable for non-Sequential parents; use Identity for Sequential
                            try:
                                if isinstance(parent, nn.Sequential):
                                    try:
                                        idx = int(last)
                                        parent._modules[str(idx)] = nn.Identity()
                                    except Exception:
                                        setattr(parent, last, nn.Identity())
                                    print(f"Could not remove Softmax module '{target}'; replaced with Identity() instead.")
                                else:
                                    setattr(parent, last, _PassthroughCallable())
                                    print(f"Could not remove Softmax module '{target}'; left a passthrough callable on the attribute.")
                            except Exception:
                                print(f"Failed to remove or replace Softmax module '{target}'.")

        export_name = args.export_name or args.input_path.stem

        if args.export_linear_only:
            linear_mod, base_name = _build_linear_module_from_state(state_dict)
            if linear_mod is None:
                print("Could not detect a final linear layer in the weights; aborting linear-only export.")
                return
            print(f"Detected final linear layer base name: {base_name}; exporting linear-only model.")
            model_obj = linear_mod
            # replace state_dict with the linear's state dict for clarity
            state_dict = {k: v for k, v in model_obj.state_dict().items()}

        onnx_input_names = [n.strip() for n in args.onnx_input_names.split(',') if n.strip()]
        onnx_output_names = [n.strip() for n in args.onnx_output_names.split(',') if n.strip()]

        # Optionally save detailed layer info to JSON (includes output shapes if example input provided)
        if args.layer_json is not None:
            try:
                example_input = torch.randn((args.trace_batch_size, *_parse_trace_shape(args.trace_shape)))
            except Exception:
                example_input = None
            layer_info = _collect_layer_info(model_obj, example_input)
            args.layer_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.layer_json, 'w', encoding='utf-8') as jf:
                json.dump(layer_info, jf, indent=2)
            print(f"Saved layer JSON: {args.layer_json}")

        _save_exports(
            state_dict=state_dict,
            output_dir=args.output_dir,
            export_name=export_name,
            source_model_path=model_path,
            source_weights_path=weights_path,
            model_obj=model_obj,
            save_full_model=args.save_full_model,
            export_torchscript=args.export_torchscript,
            torchscript_method=args.torchscript_method,
            trace_shape=args.trace_shape,
            trace_batch_size=args.trace_batch_size,
            export_onnx=args.export_onnx,
            onnx_opset=args.onnx_opset,
            onnx_dynamic=args.onnx_dynamic,
            onnx_input_names=onnx_input_names,
            onnx_output_names=onnx_output_names,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
