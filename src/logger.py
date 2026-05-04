import threading
import logging
from typing import Dict

USAGE_KEY_ALIASES = {
    "prompt_tokens": ("prompt_tokens", "input_tokens", "request_tokens"),
    "completion_tokens": ("completion_tokens", "output_tokens", "response_tokens"),
    "total_tokens": ("total_tokens", "usage_tokens"),
}

_ALIASED_KEYS = {alias for aliases in USAGE_KEY_ALIASES.values() for alias in aliases}


def _zero_token_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _coerce_token_value(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_usage(src: Dict[str, int]) -> Dict[str, int]:
    """
    Normalize usage dictionary to ensure canonical keys are present and numeric.
    Supports alternate key names returned by different providers.
    """
    normalized: Dict[str, int] = {}

    for canonical_key, aliases in USAGE_KEY_ALIASES.items():
        for alias in aliases:
            if alias not in src:
                continue
            value = _coerce_token_value(src.get(alias))
            if value:
                normalized[canonical_key] = normalized.get(canonical_key, 0) + value
                break

    if "total_tokens" not in normalized:
        prompt_total = normalized.get("prompt_tokens", 0)
        completion_total = normalized.get("completion_tokens", 0)
        if prompt_total or completion_total:
            normalized["total_tokens"] = prompt_total + completion_total

    for key, value in src.items():
        if key in _ALIASED_KEYS:
            continue
        coerced_value = _coerce_token_value(value)
        if coerced_value:
            normalized[key] = normalized.get(key, 0) + coerced_value

    return normalized


def _merge_usage(dest: Dict[str, int], src: Dict[str, int]) -> None:
    if not src:
        return

    normalized_src = _normalize_usage(src)

    for key, value in normalized_src.items():
        dest[key] = dest.get(key, 0) + value


def initialize_logger(log_path, logger_name=None):
    if logger_name is None:
        logger_name = threading.current_thread().name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger

