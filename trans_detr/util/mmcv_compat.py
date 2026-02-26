import os.path as osp

import torch


def list_from_file(filename, encoding="utf-8"):
    """Read a text file into a list of stripped lines.

    This provides the small subset of ``mmcv.list_from_file`` used by the
    project datasets.
    """
    if not osp.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist")

    with open(filename, "r", encoding=encoding) as f:
        return [line.rstrip("\n\r") for line in f]


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """Load a checkpoint into a model.

    Compatible with the subset of ``mmcv.runner.load_checkpoint`` used by
    ``pvt_v2.py``.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if logger is not None:
        if missing_keys:
            logger.warning("Missing keys when loading checkpoint: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected_keys)
    return checkpoint
