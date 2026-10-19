"""N01: adaptive spatio-temporal evidence integration.

``config``/``state``/``support``/``motion``/``model``/``synthetic`` implement the
support-domain-reorganisable multi-timescale measurement state field.  Importing
this package must not create a model or touch a device; heavy imports are local
to the modules that need them, and torch is an optional (``encoder``) extra.
"""

from __future__ import annotations

__all__ = ["N01Config", "load_config", "config_fingerprint"]


def __getattr__(name: str):
    if name in {"N01Config", "load_config", "config_fingerprint"}:
        from .config import N01Config, config_fingerprint, load_config

        return {
            "N01Config": N01Config,
            "load_config": load_config,
            "config_fingerprint": config_fingerprint,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
