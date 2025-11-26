"""
Director: Metacognitive Monitor + Search Engine

The Director is the core innovation of RAA. It:
1. Monitors entropy to detect "clashes" (confusion states)
2. Triggers search in the Manifold for alternative framings
3. Updates the Pointer with new goals when better framings are found
4. Provides sheaf-theoretic diagnostics for topological stuck-state detection

Implementation follows the staged approach from SEARCH_MECHANISM_DESIGN.md:
- Phase 1 (MVP): k-NN search
- Phase 2: Semantic-guided energy search
- Phase 3: Learned adaptive policy

NEW: Sheaf diagnostics provide principled topological analysis based on
"Sheaf Cohomology of Linear Predictive Coding Networks" (Seely, 2025).
"""

from .director_core import DirectorConfig, DirectorMVP
from .entropy_monitor import EntropyMonitor, compute_entropy
from .search_mvp import SearchResult, knn_search
from .sheaf_diagnostics import (
    AttentionSheafAnalyzer,
    CognitiveTopology,
    CohomologyResult,
    HodgeDecomposition,
    MonodromyAnalysis,
    SheafAnalyzer,
    SheafConfig,
    SheafDiagnostics,
    create_supervision_target,
)

__all__ = [
    # Core Director
    "EntropyMonitor",
    "compute_entropy",
    "knn_search",
    "SearchResult",
    "DirectorMVP",
    "DirectorConfig",
    "Director",
    # Sheaf Diagnostics
    "SheafAnalyzer",
    "SheafConfig",
    "SheafDiagnostics",
    "CohomologyResult",
    "HodgeDecomposition",
    "MonodromyAnalysis",
    "CognitiveTopology",
    "AttentionSheafAnalyzer",
    "create_supervision_target",
    # Matrix Monitor
    "MatrixMonitor",
    "MatrixMonitorConfig",
]


# Alias for main interface
class Director(DirectorMVP):
    """
    Main interface for the Director component.

    The Director monitors the Processor's output entropy and triggers
    search in the Manifold when confusion is detected.
    """
    pass
