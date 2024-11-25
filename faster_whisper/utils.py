import logging
import os
import re

from typing import List, Optional

import huggingface_hub
import requests

from tqdm.auto import tqdm

import json
import itertools

_HACKS = {}

def hook_alignment_heads(num_layers: int, num_heads: int):
    # Fix issue https://github.com/SYSTRAN/faster-whisper/issues/688
    # (alignment_heads are not properly set in the config.json file)
    return lambda dirname: check_alignment_heads(dirname, num_layers, num_heads)

def check_alignment_heads(dirname: str, num_layers: int, num_heads: int):
    config = os.path.join(dirname, "config.json")
    if os.path.exists(config):
        with open(config, "r", encoding="utf-8") as f:
            data = json.load(f)
        alignment_heads = data["alignment_heads"]
        max_layers = max(h for h, _ in alignment_heads)
        max_heads = max(h for _, h in alignment_heads)
        if max_layers > num_layers - 1 or max_heads > num_heads - 1:
            get_logger().warning(f"Invalid alignment heads in {config}, fixing it")
            alignment_heads = list(
                list(t) for t in itertools.product(
                    range(num_layers // 2, num_layers),
                    range(num_heads),
                )
            )
            data["alignment_heads"] = alignment_heads
            with open(config, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
}

# Equivalent models in Hugging Face
for model, v in list(_MODELS.items()):
    if model == "large-v1":
        model = "large"
    elif model == "large":
        continue
    _MODELS[f"openai/whisper-{model}"] = v

# English distilled models, with their equivalent in Hugging Face
distilled_models_en = {
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
for k, v in list(distilled_models_en.items()):
    distilled_models_en[f"distil-whisper/{k}"] = v
_MODELS.update(distilled_models_en)

# French finetuned & distilled models
for num_layers in 2, 4, 8, 16:
    model = f"whisper-large-v3-french-distil-dec{num_layers}"
    repo = f"bofenghuang/whisper-large-v3-french-distil-dec{num_layers}/ctranslate2"
    _MODELS[model] = repo
    _MODELS[f"bofenghuang/{model}"] = repo
    # See https://huggingface.co/bofenghuang/whisper-large-v3-french-distil-dec2/discussions/1
    _HACKS[repo] = hook_alignment_heads(num_layers, 20)
for model in "whisper-large-v3-french", "whisper-large-v2-french", "whisper-medium-french", :
    repo = f"bofenghuang/{model}/ctranslate2"
    _MODELS[model] = repo
    _MODELS[f"bofenghuang/{model}"] = repo


def available_models() -> List[str]:
    """Returns the names of available models."""
    return list(_MODELS.keys())


def get_assets_path():
    """Returns the path to the assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_logger():
    """Returns the module logger."""
    return logging.getLogger("faster_whisper")


def download_model(
    size_or_id: str,
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
):
    """Downloads a CTranslate2 Whisper model from the Hugging Face Hub.

    Args:
      size_or_id: Size of the model to download from https://huggingface.co/Systran
        (tiny, tiny.en, base, base.en, small, small.en, distil-small.en, medium, medium.en,
        distil-medium.en, large-v1, large-v2, large-v3, large, distil-large-v2,
        distil-large-v3), or a CTranslate2-converted model ID from the Hugging Face Hub
        (e.g. Systran/faster-whisper-large-v3).
      output_dir: Directory where the model should be saved. If not set, the model is saved in
        the cache directory.
      local_files_only:  If True, avoid downloading the file and return the path to the local
        cached file if it exists.
      cache_dir: Path to the folder where cached files are stored.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """
    if re.match(r".*/.*", size_or_id) and size_or_id not in _MODELS:
        repo_id = size_or_id
        subfolder = None
        model_path_post_hook = None
    else:
        repo_id = _MODELS.get(size_or_id)
        if repo_id is None:
            raise ValueError(
                "Invalid model size '%s', expected one of: %s"
                % (size_or_id, ", ".join(_MODELS.keys()))
            )
        model_path_post_hook = _HACKS.get(repo_id)
        folders = repo_id.split("/")
        if len(folders) == 2:
            repo_id, subfolder = repo_id, None
        else:
            repo_id, subfolder = "/".join(folders[:2]), "/".join(folders[2:])

    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]
    if subfolder:
        allow_patterns = [f"{subfolder}/{p}" for p in allow_patterns]

    kwargs = {
        "local_files_only": local_files_only,
        "allow_patterns": allow_patterns,
        "tqdm_class": disabled_tqdm,
    }

    if output_dir is not None:
        kwargs["local_dir"] = output_dir
        kwargs["local_dir_use_symlinks"] = False

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    try:
        model_path = huggingface_hub.snapshot_download(repo_id, **kwargs)
    except (
        huggingface_hub.utils.HfHubHTTPError,
        requests.exceptions.ConnectionError,
    ) as exception:
        logger = get_logger()
        logger.warning(
            "An error occured while synchronizing the model %s from the Hugging Face Hub:\n%s",
            repo_id,
            exception,
        )
        logger.warning(
            "Trying to load the model directly from the local cache, if it exists."
        )

        kwargs["local_files_only"] = True
        model_path = huggingface_hub.snapshot_download(repo_id, **kwargs)

    if subfolder:
        model_path = os.path.join(model_path, subfolder)

    if model_path_post_hook:
        model_path_post_hook(model_path)

    return model_path

def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class disabled_tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )
