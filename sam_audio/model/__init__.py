# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from importlib import import_module


_EXPORT_TO_MODULE = {
    "SAMAudio": ".model",
    "SAMAudioJudgeModel": ".judge",
    "SAMAudioJudgeOutput": ".judge",
}

__all__ = list(_EXPORT_TO_MODULE)


def __getattr__(name):
    if name not in _EXPORT_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
