import json
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, found {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = self.lora_up(self.dropout(self.lora_down(x)))
        return base + update * self.scaling


def freeze_module_parameters(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def _matches_prefix(name: str, prefixes: Optional[Iterable[str]]) -> bool:
    if prefixes is None:
        return True
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _matches_target(name: str, target_names: Optional[Iterable[str]]) -> bool:
    if target_names is None:
        return True
    return any(name == target or name.endswith(f".{target}") for target in target_names)


def find_linear_module_names(
    module: nn.Module,
    prefixes: Optional[Iterable[str]] = None,
    target_names: Optional[Iterable[str]] = None,
) -> list[str]:
    result = []
    for name, child in module.named_modules():
        if not name:
            continue
        if not isinstance(child, nn.Linear):
            continue
        if not _matches_prefix(name, prefixes):
            continue
        if not _matches_target(name, target_names):
            continue
        result.append(name)
    return result


def _split_parent(module_name: str) -> tuple[str, str]:
    parts = module_name.split(".")
    return ".".join(parts[:-1]), parts[-1]


def _get_submodule(module: nn.Module, module_name: str) -> nn.Module:
    if not module_name:
        return module
    current = module
    for part in module_name.split("."):
        current = getattr(current, part)
    return current


def apply_lora_to_module_names(
    module: nn.Module,
    module_names: list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    applied = []
    for module_name in module_names:
        parent_name, child_name = _split_parent(module_name)
        parent = _get_submodule(module, parent_name)
        child = getattr(parent, child_name)
        if isinstance(child, LoRALinear):
            applied.append(module_name)
            continue
        if not isinstance(child, nn.Linear):
            raise ValueError(f"Expected nn.Linear at {module_name}, found {type(child)}")
        setattr(
            parent,
            child_name,
            LoRALinear(
                base_layer=child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        applied.append(module_name)
    return applied


def apply_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    prefixes: Optional[Iterable[str]] = None,
    target_names: Optional[Iterable[str]] = None,
) -> list[str]:
    module_names = find_linear_module_names(
        module,
        prefixes=prefixes,
        target_names=target_names,
    )
    return apply_lora_to_module_names(
        module,
        module_names=module_names,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def extract_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state = module.state_dict()
    return {
        key: value.cpu()
        for key, value in state.items()
        if ".lora_down." in key or ".lora_up." in key
    }


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in module.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def save_lora_adapter(
    module: nn.Module,
    output_dir: str | Path,
    *,
    checkpoint_path: str,
    module_names: list[str],
    rank: int,
    alpha: float,
    dropout: float,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "base_checkpoint": checkpoint_path,
        "module_names": module_names,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
    }
    with open(output_dir / "adapter_config.json", "w") as fout:
        json.dump(config, fout, indent=2, sort_keys=True)

    torch.save(extract_lora_state_dict(module), output_dir / "adapter_model.pt")


def load_lora_adapter(
    module: nn.Module,
    adapter_dir: str | Path,
    map_location: str | torch.device = "cpu",
):
    adapter_dir = Path(adapter_dir)
    with open(adapter_dir / "adapter_config.json") as fin:
        config = json.load(fin)

    apply_lora_to_module_names(
        module,
        module_names=config["module_names"],
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=float(config["dropout"]),
    )
    state = torch.load(adapter_dir / "adapter_model.pt", map_location=map_location)
    load_result = module.load_state_dict(state, strict=False)
    if load_result is None:
        missing, unexpected = [], []
    else:
        missing, unexpected = load_result
    unexpected = [
        key for key in unexpected if ".lora_down." in key or ".lora_up." in key
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys while loading adapter: {unexpected}")
    missing = [key for key in missing if ".lora_down." in key or ".lora_up." in key]
    if missing:
        raise RuntimeError(f"Missing LoRA keys while loading adapter: {missing}")
    return config
