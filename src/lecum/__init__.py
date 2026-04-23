"""LECUM package.

Para evitar efeitos colaterais de import em ambientes mínimos,
este módulo não importa submódulos pesados automaticamente.
"""

__all__ = [
    "models",
    "training",
    "analysis",
    "data",
    "strategy",
    "allocation",
    "config",
    "logging_utils",
]
