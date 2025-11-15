"""
Director: Metacognitive Monitor + Search Engine

The Director is the core innovation of RAA. It:
1. Monitors entropy to detect "clashes" (confusion states)
2. Triggers search in the Manifold for alternative framings
3. Updates the Pointer with new goals when better framings are found

Implementation follows the staged approach from SEARCH_MECHANISM_DESIGN.md:
- Phase 1 (MVP): k-NN search
- Phase 2: Semantic-guided energy search
- Phase 3: Learned adaptive policy
"""

from .director_core import DirectorConfig, DirectorMVP
from .entropy_monitor import EntropyMonitor, compute_entropy
from .search_mvp import SearchResult, knn_search

__all__ = [
    "EntropyMonitor",
    "compute_entropy",
    "knn_search",
    "SearchResult",
    "DirectorMVP",
    "DirectorConfig",
    "Director",
]


# Alias for main interface
class Director(DirectorMVP):
    """
    Main interface for the Director component.

    The Director monitors the Processor's output entropy and triggers
    search in the Manifold when confusion is detected.
    """
    pass
