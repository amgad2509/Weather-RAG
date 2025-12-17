# src/api/routes/__init__.py

from .base_route import router
from . import chat as _chat  # noqa: F401 (register routes)
