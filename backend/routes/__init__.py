"""HTTP route blueprints."""

from .documents import documents_bp
from .search import search_bp
from .system import system_bp

__all__ = ["documents_bp", "search_bp", "system_bp"]
